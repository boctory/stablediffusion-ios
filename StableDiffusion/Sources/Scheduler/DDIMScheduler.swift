import CoreML
import Foundation

public class DDIMScheduler {
    private let betaStart: Float = 0.00085
    private let betaEnd: Float = 0.012
    private let trainTimesteps: Int = 1000
    private let inferenceSteps: Int = 50
    
    private(set) var timesteps: [Float]
    private var alphas: [Float]
    private var alphasCumprod: [Float]
    private var timestepMap: [Int: Int]
    
    public init() {
        // Initialize arrays first
        alphas = []
        alphasCumprod = []
        timesteps = []
        timestepMap = [:]
        
        // Generate betas and calculate alphas
        let betas = (0..<trainTimesteps).map { step in
            betaStart + Float(step) * (betaEnd - betaStart) / Float(trainTimesteps - 1)
        }
        alphas = betas.map { 1.0 - $0 }
        
        // Calculate cumulative products of alphas
        alphasCumprod = alphas.reduce(into: []) { result, alpha in
            if result.isEmpty {
                result.append(alpha)
            } else {
                result.append(result.last! * alpha)
            }
        }
        
        // Set timesteps for inference
        let stepRatio = trainTimesteps / inferenceSteps
        timesteps = (0..<inferenceSteps).map { step in
            Float(trainTimesteps - 1 - step * stepRatio)
        }
        
        // Create timestep mapping
        for (index, timestep) in timesteps.enumerated() {
            timestepMap[Int(timestep)] = index
        }
    }
    
    public func step(modelOutput: MLMultiArray, timestep: Float, sample: MLMultiArray) -> MLMultiArray {
        let prevTimestep = timestep > 0 ? timestep - 1 : 0
        let alphaProdTPrev = alphasCumprod[Int(prevTimestep)]
        let betaProdTPrev = 1 - alphaProdTPrev
        
        // Compute predicted original sample from predicted noise
        let predOriginalSample = try! computePredictedOriginalSample(
            sample: sample,
            noise: modelOutput,
            timestep: timestep
        )
        
        // Compute coefficient for noise
        let noiseCoefficient = sqrt(betaProdTPrev)
        
        // Compute previous sample
        return try! computePreviousSample(
            predOriginalSample: predOriginalSample,
            noise: modelOutput,
            timestep: timestep,
            noiseCoefficient: noiseCoefficient
        )
    }
    
    private func computePredictedOriginalSample(sample: MLMultiArray, noise: MLMultiArray, timestep: Float) throws -> MLMultiArray {
        let alphaProd = alphasCumprod[Int(timestep)]
        let betaProd = 1 - alphaProd
        
        // x_0 = (x_t - sqrt(beta_t) * noise) / sqrt(alpha_t)
        let result = try MLMultiArray(shape: sample.shape, dataType: .float32)
        
        for i in 0..<sample.count {
            let sampleValue = sample[i].floatValue
            let noiseValue = noise[i].floatValue
            let value = (sampleValue - sqrt(betaProd) * noiseValue) / sqrt(alphaProd)
            result[i] = NSNumber(value: value)
        }
        
        return result
    }
    
    private func computePreviousSample(predOriginalSample: MLMultiArray, noise: MLMultiArray, timestep: Float, noiseCoefficient: Float) throws -> MLMultiArray {
        let alphaProdTPrev = alphasCumprod[Int(timestep > 0 ? timestep - 1 : 0)]
        
        // x_{t-1} = sqrt(alpha_{t-1}) * x_0 + sqrt(beta_{t-1}) * noise
        let result = try MLMultiArray(shape: predOriginalSample.shape, dataType: .float32)
        
        for i in 0..<predOriginalSample.count {
            let x0Value = predOriginalSample[i].floatValue
            let noiseValue = noise[i].floatValue
            let value = sqrt(alphaProdTPrev) * x0Value + noiseCoefficient * noiseValue
            result[i] = NSNumber(value: value)
        }
        
        return result
    }
} 