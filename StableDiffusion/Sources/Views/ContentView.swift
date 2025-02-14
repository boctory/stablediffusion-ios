import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel = GenerationViewModel()
    
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                ScrollView {
                    VStack(spacing: 20) {
                        // Image display area
                        if let image = viewModel.generatedImage {
                            Image(uiImage: image)
                                .resizable()
                                .aspectRatio(contentMode: .fit)
                                .frame(maxWidth: .infinity)
                                .frame(height: 512)
                                .cornerRadius(12)
                                .shadow(radius: 8)
                        } else {
                            RoundedRectangle(cornerRadius: 12)
                                .fill(Color.gray.opacity(0.2))
                                .frame(height: 512)
                                .overlay(
                                    Image(systemName: "photo")
                                        .font(.largeTitle)
                                        .foregroundColor(.gray)
                                )
                        }
                        
                        // Input area
                        VStack(spacing: 16) {
                            TextField("Enter your prompt...", text: $viewModel.prompt)
                                .textFieldStyle(RoundedBorderTextFieldStyle())
                                .padding(.horizontal)
                            
                            // Style selection
                            VStack(alignment: .leading, spacing: 8) {
                                Text("Style")
                                    .font(.headline)
                                    .padding(.horizontal)
                                
                                ScrollView(.horizontal, showsIndicators: false) {
                                    HStack(spacing: 12) {
                                        ForEach(viewModel.styles, id: \.id) { style in
                                            StyleButton(
                                                label: style.label,
                                                isSelected: viewModel.selectedStyle == style.preset,
                                                action: { viewModel.selectedStyle = style.preset }
                                            )
                                        }
                                    }
                                    .padding(.horizontal)
                                }
                            }
                            
                            // Generate button
                            Button(action: viewModel.generateImage) {
                                if viewModel.isGenerating {
                                    ProgressView()
                                        .progressViewStyle(CircularProgressViewStyle())
                                } else {
                                    Text("Generate Image")
                                        .frame(maxWidth: .infinity)
                                }
                            }
                            .buttonStyle(PrimaryButtonStyle())
                            .disabled(viewModel.prompt.isEmpty || viewModel.isGenerating)
                            .padding(.horizontal)
                            
                            if let error = viewModel.error {
                                Text(error)
                                    .foregroundColor(.red)
                                    .font(.caption)
                                    .padding(.horizontal)
                            }
                        }
                    }
                    .padding()
                }
            }
            .navigationTitle("Diffusion 203")
        }
    }
}

struct StyleButton: View {
    let label: String
    let isSelected: Bool
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            Text(label)
                .padding(.horizontal, 16)
                .padding(.vertical, 8)
                .background(isSelected ? Color.blue : Color.gray.opacity(0.2))
                .foregroundColor(isSelected ? .white : .primary)
                .cornerRadius(20)
        }
    }
}

struct PrimaryButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .padding()
            .background(Color.blue)
            .foregroundColor(.white)
            .cornerRadius(8)
            .opacity(configuration.isPressed ? 0.8 : 1.0)
    }
}

class GenerationViewModel: ObservableObject {
    @Published var prompt: String = ""
    @Published var selectedStyle: String = "photographic"
    @Published var generatedImage: UIImage?
    @Published var isGenerating = false
    @Published var error: String?
    
    let styles = [
        Style(id: "natural", label: "Natural", preset: "photographic"),
        Style(id: "artificial", label: "Artificial", preset: "digital-art"),
        Style(id: "real", label: "Real", preset: "enhance"),
        Style(id: "comics", label: "Comics", preset: "comic-book")
    ]
    
    func generateImage() {
        guard !prompt.isEmpty else { return }
        
        isGenerating = true
        error = nil
        
        let API_KEY = "sk-TgaYBMetnisVcLXyfbBFsYzDaoj4xrHupLppdOkGtYXhOKUV"
        let API_HOST = "https://api.stability.ai"
        
        // Create URL request
        guard let url = URL(string: "\(API_HOST)/v1/generation/stable-diffusion-v1-6/text-to-image") else {
            error = "Invalid URL"
            isGenerating = false
            return
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.setValue("Bearer \(API_KEY)", forHTTPHeaderField: "Authorization")
        
        let parameters: [String: Any] = [
            "text_prompts": [
                ["text": prompt, "weight": 1]
            ],
            "cfg_scale": 7,
            "height": 512,
            "width": 512,
            "samples": 1,
            "steps": 30,
            "style_preset": selectedStyle
        ]
        
        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: parameters)
        } catch {
            self.error = "Failed to create request"
            isGenerating = false
            return
        }
        
        URLSession.shared.dataTask(with: request) { [weak self] data, response, error in
            DispatchQueue.main.async {
                self?.isGenerating = false
                
                if let error = error {
                    self?.error = error.localizedDescription
                    return
                }
                
                guard let data = data else {
                    self?.error = "No data received"
                    return
                }
                
                do {
                    let result = try JSONDecoder().decode(GenerationResponse.self, from: data)
                    if let base64String = result.artifacts.first?.base64,
                       let imageData = Data(base64Encoded: base64String),
                       let image = UIImage(data: imageData) {
                        self?.generatedImage = image
                    } else {
                        self?.error = "Failed to decode image"
                    }
                } catch {
                    if let errorResponse = try? JSONDecoder().decode(ErrorResponse.self, from: data) {
                        self?.error = errorResponse.message
                    } else {
                        self?.error = "Failed to decode response"
                    }
                }
            }
        }.resume()
    }
}

struct Style {
    let id: String
    let label: String
    let preset: String
}

struct GenerationResponse: Codable {
    let artifacts: [Artifact]
}

struct Artifact: Codable {
    let base64: String
}

struct ErrorResponse: Codable {
    let message: String
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
} 