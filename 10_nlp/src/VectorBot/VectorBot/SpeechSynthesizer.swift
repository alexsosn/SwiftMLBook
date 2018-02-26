//
//  SpeechSynthesizer.swift
//  VectorBot
//
//  Created by Oleksandr on 6/22/17.
//  Copyright © 2017 OWL. All rights reserved.
//

import Foundation
import Speech

class SpeechSynthesizer: NSObject, AVSpeechSynthesizerDelegate {
    static var shared = SpeechSynthesizer()
    
    private var synthesizer = AVSpeechSynthesizer()
    var voice = AVSpeechSynthesisVoice(language: "en-US")
    
    public func prepare() {
        let dummyUtterance = AVSpeechUtterance(string: " ")
        dummyUtterance.voice = AVSpeechSynthesisVoice(language: "en-US")
        
        synthesizer.speak(dummyUtterance)
    }
    
    public func speakAloud(word: String) {
        if synthesizer.isSpeaking {
            synthesizer.stopSpeaking(at: .immediate)
        }
        
        let utterance = AVSpeechUtterance(string: word)
        utterance.rate = 0.4
        utterance.preUtteranceDelay = 0.1;
        utterance.postUtteranceDelay = 0.1;
        utterance.voice = self.voice
        
        
        synthesizer.speak(utterance)
    }
    
    public func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didStart utterance: AVSpeechUtterance) {
        
    }
    
    public func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        
    }
    
    public func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        
    }
}
