import XCTest
@testable import StableDiffusion

final class StableDiffusionTests: XCTestCase {
    var pipeline: StableDiffusionPipeline!
    var resourcesURL: URL!
    
    override func setUpWithError() throws {
        try super.setUpWithError()
        
        // Get the test resources URL
        guard let url = Bundle.module.url(forResource: "Resources", withExtension: nil) else {
            throw XCTSkip("Resources not available")
        }
        resourcesURL = url
        
        // Initialize pipeline
        pipeline = try StableDiffusionPipeline(resourcesURL: resourcesURL)
    }
    
    override func tearDownWithError() throws {
        pipeline = nil
        resourcesURL = nil
        try super.tearDownWithError()
    }
    
    func testTokenizer() throws {
        let tokenizer = try CLIPTokenizer()
        
        // Test basic tokenization
        let text = "a photo of a cat"
        let tokens = try tokenizer.encode(text)
        
        XCTAssertEqual(tokens.shape, [1, 77])
        XCTAssertEqual(tokens.dataType, .int32)
        
        // Test empty text
        let emptyTokens = try tokenizer.encode("")
        XCTAssertEqual(emptyTokens.shape, [1, 77])
        
        // Test long text
        let longText = String(repeating: "test ", count: 100)
        let longTokens = try tokenizer.encode(longText)
        XCTAssertEqual(longTokens.shape, [1, 77])
    }
    
    func testScheduler() {
        let scheduler = DDIMScheduler()
        
        // Test timesteps
        XCTAssertEqual(scheduler.timesteps.count, 50)
        XCTAssertGreaterThan(scheduler.timesteps[0], scheduler.timesteps[1])
        
        // Test step function
        let sampleShape = [1, 4, 64, 64] as [NSNumber]
        let sample = try! MLMultiArray(shape: sampleShape, dataType: .float32)
        let noise = try! MLMultiArray(shape: sampleShape, dataType: .float32)
        
        let result = scheduler.step(
            modelOutput: noise,
            timestep: scheduler.timesteps[0],
            sample: sample
        )
        
        XCTAssertEqual(result.shape, sampleShape)
        XCTAssertEqual(result.dataType, .float32)
    }
    
    func testPipeline() async throws {
        let configuration = StableDiffusionPipeline.Configuration(
            steps: 20,
            width: 512,
            height: 512,
            seed: 42
        )
        
        let image = try await pipeline.generateImage(
            prompt: "a photo of a cat",
            configuration: configuration
        )
        
        XCTAssertNotNil(image)
        XCTAssertEqual(image.width, 512)
        XCTAssertEqual(image.height, 512)
    }
    
    func testPipelineWithNegativePrompt() async throws {
        let configuration = StableDiffusionPipeline.Configuration(
            steps: 20,
            width: 512,
            height: 512,
            seed: 42
        )
        
        let image = try await pipeline.generateImage(
            prompt: "a photo of a cat",
            negativePrompt: "blurry, bad quality",
            configuration: configuration
        )
        
        XCTAssertNotNil(image)
        XCTAssertEqual(image.width, 512)
        XCTAssertEqual(image.height, 512)
    }
    
    func testPipelineWithDifferentSizes() async throws {
        let sizes = [(256, 256), (512, 512), (768, 512), (512, 768)]
        
        for (width, height) in sizes {
            let configuration = StableDiffusionPipeline.Configuration(
                steps: 20,
                width: width,
                height: height,
                seed: 42
            )
            
            let image = try await pipeline.generateImage(
                prompt: "a photo of a cat",
                configuration: configuration
            )
            
            XCTAssertNotNil(image)
            XCTAssertEqual(image.width, width)
            XCTAssertEqual(image.height, height)
        }
    }
    
    func testPipelinePerformance() async throws {
        let configuration = StableDiffusionPipeline.Configuration(
            steps: 20,
            width: 512,
            height: 512,
            seed: 42
        )
        
        measure {
            let expectation = XCTestExpectation(description: "Generate image")
            
            Task {
                do {
                    let _ = try await pipeline.generateImage(
                        prompt: "a photo of a cat",
                        configuration: configuration
                    )
                    expectation.fulfill()
                } catch {
                    XCTFail("Failed to generate image: \(error)")
                }
            }
            
            wait(for: [expectation], timeout: 60.0)
        }
    }
} 