import CoreML
import Foundation
import CoreImage
import Accelerate

public class StableDiffusionPipeline {
    private let textEncoder: MLModel
    private let unet: MLModel
    private let decoder: MLModel
    private let tokenizer: CLIPTokenizer
    private let scheduler: DDIMScheduler
    
    private let queue = DispatchQueue(label: "com.stablediffusion.pipeline", qos: .userInitiated)
    private let processingQueue = OperationQueue()
    
    public struct Configuration {
        public let steps: Int
        public let batchSize: Int
        public let width: Int
        public let height: Int
        public let seed: UInt32
        
        public init(steps: Int = 50, batchSize: Int = 1, width: Int = 512, height: Int = 512, seed: UInt32 = 0) {
            self.steps = steps
            self.batchSize = batchSize
            self.width = width
            self.height = height
            self.seed = seed
        }
    }
    
    public init(resourcesURL: URL) throws {
        // Initialize models with configuration
        let config = MLModelConfiguration()
        config.computeUnits = .all
        config.allowLowPrecisionAccumulationOnGPU = true
        
        // Initialize models
        let textEncoderURL = resourcesURL.appendingPathComponent("TextEncoder.mlpackage")
        let unetURL = resourcesURL.appendingPathComponent("Unet.mlpackage")
        let decoderURL = resourcesURL.appendingPathComponent("Decoder.mlpackage")
        
        self.textEncoder = try MLModel(contentsOf: textEncoderURL, configuration: config)
        self.unet = try MLModel(contentsOf: unetURL, configuration: config)
        self.decoder = try MLModel(contentsOf: decoderURL, configuration: config)
        
        // Initialize tokenizer and scheduler
        self.tokenizer = try CLIPTokenizer()
        self.scheduler = DDIMScheduler()
        
        // Configure processing queue
        processingQueue.maxConcurrentOperationCount = 1
        processingQueue.qualityOfService = .userInitiated
    }
    
    public func generateImage(prompt: String, negativePrompt: String = "", configuration: Configuration = Configuration()) async throws -> CGImage {
        // 1. Encode text prompt (can be done in parallel)
        async let encodedPromptTask = encodeText(prompt)
        async let encodedNegativePromptTask = encodeText(negativePrompt)
        
        let (encodedPrompt, encodedNegativePrompt) = try await (encodedPromptTask, encodedNegativePromptTask)
        
        // 2. Generate initial noise
        var latents = generateRandomLatents(
            width: configuration.width,
            height: configuration.height,
            batchSize: configuration.batchSize,
            seed: configuration.seed
        )
        
        // 3. Setup progress
        let timesteps = scheduler.timesteps
        
        // 4. Denoising loop with optimized processing
        for step in 0..<configuration.steps {
            let timestep = timesteps[step]
            
            // Predict noise residual
            let noiseResidual = try await queue.asyncResult {
                try self.predictNoiseResidual(
                    latents: latents,
                    timestep: timestep,
                    encodedPrompt: encodedPrompt,
                    encodedNegativePrompt: encodedNegativePrompt
                )
            }
            
            // Scheduler step (optimized with vDSP)
            latents = try await queue.asyncResult {
                self.scheduler.step(
                    modelOutput: noiseResidual,
                    timestep: timestep,
                    sample: latents
                )
            }
        }
        
        // 5. Decode latents to image
        return try await queue.asyncResult {
            try self.decodeLatentsToImage(latents)
        }
    }
    
    private func encodeText(_ text: String) async throws -> MLMultiArray {
        try await queue.asyncResult {
            let tokens = try self.tokenizer.encode(text)
            let input = try MLDictionaryFeatureProvider(dictionary: ["input_ids": tokens])
            let output = try self.textEncoder.prediction(from: input)
            return output.featureValue(for: "last_hidden_state")!.multiArrayValue!
        }
    }
    
    private func generateRandomLatents(width: Int, height: Int, batchSize: Int, seed: UInt32) -> MLMultiArray {
        let latentHeight = height / 8
        let latentWidth = width / 8
        
        var generator: RandomNumberGenerator = SystemRandomNumberGenerator()
        if seed != 0 {
            generator = SeededRandomNumberGenerator(seed: seed)
        }
        
        let shape = [batchSize as NSNumber, 4, latentHeight as NSNumber, latentWidth as NSNumber]
        let count = shape.reduce(1) { $0 * $1.intValue }
        
        // Use vDSP for faster random number generation
        var randomValues = [Float](repeating: 0, count: count)
        for i in 0..<count {
            randomValues[i] = Float.random(in: -1...1, using: &generator)
        }
        
        let latents = try! MLMultiArray(shape: shape, dataType: .float32)
        let ptr = UnsafeMutablePointer<Float>(OpaquePointer(latents.dataPointer))
        memcpy(ptr, &randomValues, count * MemoryLayout<Float>.size)
        
        return latents
    }
    
    private func predictNoiseResidual(latents: MLMultiArray, timestep: Float, encodedPrompt: MLMultiArray, encodedNegativePrompt: MLMultiArray) throws -> MLMultiArray {
        let timestepArray = try MLMultiArray(shape: [1] as [NSNumber], dataType: .float32)
        timestepArray[0] = NSNumber(value: timestep)
        
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "sample": latents,
            "timestep": timestepArray,
            "encoder_hidden_states": encodedPrompt
        ])
        
        let output = try unet.prediction(from: input)
        return output.featureValue(for: "noise_pred")!.multiArrayValue!
    }
    
    private func decodeLatentsToImage(_ latents: MLMultiArray) throws -> CGImage {
        let input = try MLDictionaryFeatureProvider(dictionary: ["latent": latents])
        let output = try decoder.prediction(from: input)
        guard let pixelBuffer = output.featureValue(for: "image")?.imageBufferValue else {
            throw NSError(domain: "StableDiffusionPipeline", code: -1, userInfo: [NSLocalizedDescriptionKey: "Failed to get image buffer"])
        }
        
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext(options: nil)
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else {
            throw NSError(domain: "StableDiffusionPipeline", code: -1, userInfo: [NSLocalizedDescriptionKey: "Failed to convert to CGImage"])
        }
        return cgImage
    }
}

extension DispatchQueue {
    func asyncResult<T>(execute work: @escaping () throws -> T) async throws -> T {
        try await withCheckedThrowingContinuation { continuation in
            self.async {
                do {
                    let result = try work()
                    continuation.resume(returning: result)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
}

// Seeded random number generator for reproducible results
struct SeededRandomNumberGenerator: RandomNumberGenerator {
    private var generator: UInt64
    
    init(seed: UInt32) {
        generator = UInt64(seed)
    }
    
    mutating func next() -> UInt64 {
        generator = generator &* 6364136223846793005 &+ 1442695040888963407
        return generator
    }
} 

