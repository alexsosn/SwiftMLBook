//
//  NLPPreprocessor.swift
//  AssociationBot
//
//  Created by Oleksandr on 6/23/17.
//  Copyright © 2017 OWL. All rights reserved.
//

import Foundation

extension String {
    func range(from nsRange: NSRange) -> Range<String.Index>? {
        guard
            let from16 = utf16.index(utf16.startIndex, offsetBy: nsRange.location, limitedBy: utf16.endIndex),
            let to16 = utf16.index(utf16.startIndex, offsetBy: nsRange.location + nsRange.length, limitedBy: utf16.endIndex),
            let from = from16.samePosition(in: self),
            let to = to16.samePosition(in: self)
            else { return nil }
        return from ..< to
    }
}

enum NLPPreprocessorError: Error {
    case nonEnglishLanguage
}

struct NLPPreprocessor {
    
    static func preprocess(inputString: String, errorCallback: (NLPPreprocessorError)->()) -> [String] {
        let languageDetector = NSLinguisticTagger(tagSchemes: [.language], options: 0)
        languageDetector.string = inputString
        let language = languageDetector.dominantLanguage
        
//        // "und" means that NSLinguisticTagger wasn't able to recognize the language.
//        if language != "en" && language != "und" {
//            errorCallback(.nonEnglishLanguage)
//            return []
//        }
        
        // This is a workaround to make NSLinguisticTagger's lemmatizer work with short sentences.
        let string = inputString + ". Hello, world!"
        
        let tagSchemes: [NSLinguisticTagScheme] = [.tokenType, .lemma, .lexicalClass]
        
        let options = NSLinguisticTagger.Options.omitPunctuation.rawValue | NSLinguisticTagger.Options.omitWhitespace.rawValue
        let tagger = NSLinguisticTagger(tagSchemes: NSLinguisticTagger.availableTagSchemes(forLanguage: "en"), options: Int(options))
        tagger.string = string
        let range = NSRange(location: 0, length: string.utf16.count)
        
        var resultTokens = [String?]()
        let queryOptions = NSLinguisticTagger.Options(rawValue: options)
        
//        // Save only words while keeping the total count of tokens.
//        tagger.enumerateTags(in: range,
//                             scheme: .tokenType,
//                             options: queryOptions) { (tag, range1, range2, _) in
//                                guard let tag = tag else {
//                                    resultTokens.append(nil)
//                                    return
//                                }
//                                let token = string.substring(with: string.range(from: range1)!)
//
//                                if tag == .word {
//                                    resultTokens.append(token)
//                                } else {
//                                    resultTokens.append(nil)
//                                }
//        }
//
//
//        // If word has lemma save it.
//        var i = 0
//        tagger.enumerateTags(in: range,
//                             scheme: .lemma,
//                             options: queryOptions) { (tag, range1, range2, _) in
//                                defer {i+=1}
//                                guard let tag = tag else { return }
//                                resultTokens[i] = tag.rawValue
//        }
//
//
//         Using POS tagger to remove all word types that are not playable
        let posToPreserve: Set<NSLinguisticTag> = Set([.noun, .verb, .adjective, .adverb, .interjection, .idiom, .otherWord])
        
//        i = 0
//        tagger.enumerateTags(in: range,
//                             scheme: .lexicalClass,
//                             options: queryOptions) { (tag, range1, range2, _) in
//                                defer {i+=1}
//                                guard let tag = tag else { return }
//                                if !posToPreserve.contains(tag) {
//                                    resultTokens[i] = nil
//                                }
//        }
        
        for scheme in tagSchemes {
            var i = 0
            tagger.enumerateTags(in: range, scheme: scheme, options: queryOptions)
            { (tag, range1, _, _) in
                defer { i+=1 }
                
                guard let tag = tag else {
                    // Preserve total count of tokens.
                    if scheme == .tokenType { resultTokens.append(nil) }
                    return
                }
                
                switch scheme {
                case .tokenType:
                    // Save only words while keeping the total count of tokens.
                    let token = string.substring(with: string.range(from: range1)!)
                    
                    if tag == .word {
                        resultTokens.append(token)
                    } else {
                        resultTokens.append(nil)
                    }
                case .lemma:
                    // If word has lemma save it.
                    resultTokens[i] = tag.rawValue
                case .lexicalClass:
                    // Using POS tagger to remove all word types that are not playable.
                    if !posToPreserve.contains(tag) {
                        resultTokens[i] = nil
                    }
                default:
                    break
                }
            }
        }
        // This is a workaround to make NSLinguisticTagger's lemmatizer work with short sentences.
        var result = resultTokens.flatMap{$0}
        print(result)
        result.removeLast()
        result.removeLast()
        return result
    }
}
