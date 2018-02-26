//
//  Classifier.swift
//  Empathy
//
//  Created by Alex Sosnovshchenko on 7/12/17.
//  Copyright © 2017 Stigma Inc. All rights reserved.
//

import Foundation
import Vision
import AVFoundation
import UIKit

enum FaceExpressions: String {
    case angry = "angry"
    case anxious = "anxious"
    case neutral = "neutral"
    case happy = "happy"
    case sad = "sad"
}

enum ClassifierError: Error {
    case unableToResizeBuffer
    case noResults
}

class Classifier {
    public static let shared = Classifier()
    
    private let visionModel: VNCoreMLModel
    var visionRequests = [VNRequest]()
    var completion: ((_ label: [(FaceExpressions, Double)], _ error: Error?)->())?

    private init() {
        guard let visionModel = try? VNCoreMLModel(for: Emotions_Stigma().model) else {
            fatalError("Could not load model")
        }
        
        self.visionModel = visionModel
        
        let classificationRequest = VNCoreMLRequest(model: visionModel, completionHandler: classificationResultHandler)
        classificationRequest.imageCropAndScaleOption = .centerCrop
        visionRequests = [classificationRequest]
    }
    
    public func classifyFace(image: CGImage, completion: @escaping (_ labels: [(FaceExpressions, Double)], _ error: Error?)->()) {
        self.completion = completion
        let imageRequestHandler = VNImageRequestHandler(cgImage: image, orientation: .up)
        do {
            try imageRequestHandler.perform(visionRequests)
        } catch {
            print(error)
            completion([], error)
        }
    }
    
    private func classificationResultHandler(request: VNRequest, error: Error?) {
        if let error = error {
            print(error.localizedDescription)
            self.completion?([], error)
            return
        }
        guard let results = request.results as? [VNClassificationObservation] else {
            print("No results")
            self.completion?([], ClassifierError.noResults)
            return
        }
        
        let sortedResults = results
            .sorted { $0.confidence > $1.confidence }
            .map{(FaceExpressions(rawValue:$0.identifier)!, Double($0.confidence))}

        self.completion?(sortedResults, nil)
        print(sortedResults)
    }
}
