//
//  VoiceRecognizer.swift
//  VectorBot
//
//  Created by Oleksandr on 6/22/17.
//  Copyright © 2017 OWL. All rights reserved.
//

import Foundation
import Speech

class VoiceRecognizer: NSObject, SFSpeechRecognizerDelegate {
    static var shared = VoiceRecognizer()
    
    private let speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))!
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private let audioEngine = AVAudioEngine()
    
    public var isListening: Bool {
        return audioEngine.isRunning
    }
    
    public func stopListening() {
        self.audioEngine.stop()
        self.recognitionRequest?.endAudio()
    }
    
    public func startListening(gotResultBlock: @escaping (String)->(), endBlock: @escaping (Error?)->()) {
        speechRecognizer.delegate = self
        
        // Cancel the previous task if it's running.
        if let recognitionTask = recognitionTask {
            recognitionTask.cancel()
            self.recognitionTask = nil
        }
        do {
            let audioSession = AVAudioSession.sharedInstance()
            try audioSession.setCategory(AVAudioSessionCategoryRecord)
            try audioSession.setMode(AVAudioSessionModeMeasurement)
            try audioSession.setActive(true, with: .notifyOthersOnDeactivation)
        } catch {
            print(error)
        }
        
        recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
        
        let inputNode = audioEngine.inputNode
        guard let recognitionRequest = recognitionRequest else { fatalError("Unable to created a SFSpeechAudioBufferRecognitionRequest object") }
        
        // Configure request so that results are returned before audio recording is finished
        recognitionRequest.shouldReportPartialResults = false
        
        // A recognition task represents a speech recognition session.
        // We keep a reference to the task so that it can be cancelled.
        recognitionTask = speechRecognizer.recognitionTask(with: recognitionRequest) { [weak self] result, error in
            guard let `self` = self else { return }
            var isFinal = false
            
            if let result = result {
                let string = result.bestTranscription.formattedString
                gotResultBlock(string)
                isFinal = result.isFinal
            }
            
            if error != nil || isFinal {
                print(error?.localizedDescription ?? "No errors.")
                
                self.audioEngine.stop()
                inputNode.removeTap(onBus: 0)
                
                self.recognitionRequest = nil
                self.recognitionTask = nil
                endBlock(error)
            }
        }
        
        let recordingFormat = inputNode.outputFormat(forBus: 0)
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { (buffer: AVAudioPCMBuffer, when: AVAudioTime) in
            self.recognitionRequest?.append(buffer)
        }
        
        audioEngine.prepare()
        do {
            try audioEngine.start()
        } catch {
            print(error)
        }
    }
}
