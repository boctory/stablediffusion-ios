import CoreML
import UIKit

class ModelService: ObservableObject {
    @Published var isGenerating = false
    @Published var progress: Float = 0.0
    @Published var generatedImage: UIImage?
    @Published var error: String?
    
    private var vaeModel: MLModel?
    private var tokenizer: MLModel?
    
    init() {
        loadModels()
    }
    
    private func loadModels() {
        do {
            // VAE 모델 로드
            if let vaeURL = Bundle.main.url(forResource: "sdxl_vae", withExtension: "mlpackage") {
                vaeModel = try MLModel(contentsOf: vaeURL)
            } else {
                error = "VAE 모델을 찾을 수 없습니다."
                return
            }
        } catch {
            self.error = "모델 로딩 실패: \(error.localizedDescription)"
        }
    }
    
    func generateImage(from prompt: String) {
        guard !isGenerating else { return }
        guard let vae = vaeModel else {
            self.error = "모델이 로드되지 않았습니다."
            return
        }
        
        isGenerating = true
        progress = 0.0
        error = nil
        
        // 백그라운드에서 이미지 생성
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            do {
                let config = MLModelConfiguration()
                config.computeUnits = .all
                
                // 초기 노이즈 생성
                let latentSize = (1, 4, 128, 128)
                let latents = self?.generateRandomLatents(size: latentSize)
                
                // VAE를 통한 이미지 생성
                let vaeInput = try MLDictionaryFeatureProvider(dictionary: [
                    "latent": latents as Any
                ])
                
                let vaeOutput = try vae.prediction(from: vaeInput)
                if let imageArray = vaeOutput.featureValue(for: "var_662")?.multiArrayValue {
                    // Convert MLMultiArray to CGImage
                    let width = 1024
                    let height = 1024
                    let bytesPerRow = width * 4
                    var imageData = [UInt8](repeating: 0, count: width * height * 4)
                    
                    for y in 0..<height {
                        for x in 0..<width {
                            let offset = y * width + x
                            let r = UInt8(max(0, min(255, imageArray[offset].doubleValue * 255)))
                            let g = UInt8(max(0, min(255, imageArray[offset + width * height].doubleValue * 255)))
                            let b = UInt8(max(0, min(255, imageArray[offset + 2 * width * height].doubleValue * 255)))
                            
                            let pixelOffset = (y * width + x) * 4
                            imageData[pixelOffset] = r
                            imageData[pixelOffset + 1] = g
                            imageData[pixelOffset + 2] = b
                            imageData[pixelOffset + 3] = 255
                        }
                    }
                    
                    let colorSpace = CGColorSpaceCreateDeviceRGB()
                    let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
                    
                    if let context = CGContext(data: &imageData,
                                            width: width,
                                            height: height,
                                            bitsPerComponent: 8,
                                            bytesPerRow: bytesPerRow,
                                            space: colorSpace,
                                            bitmapInfo: bitmapInfo.rawValue),
                       let cgImage = context.makeImage() {
                        let image = UIImage(cgImage: cgImage)
                        DispatchQueue.main.async {
                            self?.generatedImage = image
                            self?.isGenerating = false
                            self?.progress = 1.0
                        }
                    }
                }
            } catch {
                DispatchQueue.main.async {
                    self?.error = "이미지 생성 실패: \(error.localizedDescription)"
                    self?.isGenerating = false
                }
            }
        }
    }
    
    private func generateRandomLatents(size: (Int, Int, Int, Int)) -> MLMultiArray {
        let shape = [NSNumber(value: size.0),
                    NSNumber(value: size.1),
                    NSNumber(value: size.2),
                    NSNumber(value: size.3)]
        
        let latents = try! MLMultiArray(shape: shape, dataType: .float32)
        let count = size.0 * size.1 * size.2 * size.3
        
        for i in 0..<count {
            latents[i] = NSNumber(value: Float.random(in: -1...1))
        }
        
        return latents
    }
} 