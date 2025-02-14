import CoreML
import Foundation

public class CLIPTokenizer {
    private let vocabulary: [String: Int]
    private let maxTokens: Int = 77
    
    private let startToken: Int = 49406  // <|startoftext|>
    private let endToken: Int = 49407    // <|endoftext|>
    private let padToken: Int = 0        // <|pad|>
    
    public init() throws {
        guard let vocabURL = Bundle.main.url(forResource: "clip_vocab", withExtension: "json") else {
            throw TokenizerError.missingVocabulary
        }
        
        let vocabData = try Data(contentsOf: vocabURL)
        self.vocabulary = try JSONDecoder().decode([String: Int].self, from: vocabData)
    }
    
    public func encode(_ text: String) throws -> MLMultiArray {
        var tokens = [Int]()
        
        // Add start token
        tokens.append(startToken)
        
        // Tokenize text
        // First, clean and normalize the text
        let normalizedText = text.lowercased()
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
        
        // Split into words and subwords
        let words = normalizedText.components(separatedBy: .whitespaces)
        for word in words {
            if let token = vocabulary[word] {
                tokens.append(token)
            } else {
                // Handle unknown tokens by splitting into subwords
                let subwords = splitIntoSubwords(word)
                tokens.append(contentsOf: subwords)
            }
        }
        
        // Add end token
        tokens.append(endToken)
        
        // Pad or truncate to maxTokens
        if tokens.count < maxTokens {
            tokens.append(contentsOf: Array(repeating: padToken, count: maxTokens - tokens.count))
        } else if tokens.count > maxTokens {
            tokens = Array(tokens.prefix(maxTokens))
        }
        
        // Convert to MLMultiArray
        let shape = [1, maxTokens] as [NSNumber]
        let result = try MLMultiArray(shape: shape, dataType: .int32)
        
        for (index, token) in tokens.enumerated() {
            result[index] = NSNumber(value: token)
        }
        
        return result
    }
    
    private func splitIntoSubwords(_ word: String) -> [Int] {
        var tokens = [Int]()
        var currentWord = word
        
        while !currentWord.isEmpty {
            var longestMatch: String?
            var longestMatchToken: Int?
            
            // Try to find the longest matching subword
            for length in (1...currentWord.count).reversed() {
                let prefix = String(currentWord.prefix(length))
                if let token = vocabulary[prefix] {
                    longestMatch = prefix
                    longestMatchToken = token
                    break
                }
            }
            
            if let match = longestMatch, let token = longestMatchToken {
                tokens.append(token)
                currentWord = String(currentWord.dropFirst(match.count))
            } else {
                // If no match found, treat the first character as unknown and continue
                currentWord = String(currentWord.dropFirst())
            }
        }
        
        return tokens.isEmpty ? [endToken] : tokens
    }
}

enum TokenizerError: Error {
    case missingVocabulary
} 