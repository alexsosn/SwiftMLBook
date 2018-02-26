
import Foundation

func magnitude(series3D: [[Double]]) -> [Double] {
    var result = [Double](repeating: 0.0, count: series3D.count)
    for vector3D in series3D {
        let scalar = vector3D.map{ pow($0, 2) }.reduce(0, +)
        result.append(sqrt(scalar))
    }
    
    return result
}

final class MotionClassifier {
    private let onRecognition: (MotionType)->()
    private var classifier: kNN<[Double], MotionType>

    private var motionInput: MotionInput?
    private var currentMotionType: MotionType?
    
    private(set) var isRecognizing = false
    
    init(callback: @escaping (MotionType) -> ()) {
        self.onRecognition = callback
        classifier = kNN(k: 1, distanceMetric: DTW.distance(w: 3))
    }
    
    func changeMotionType(newType: MotionType) {
        if isRecognizing {
            fatalError("Should not be called if not in training mode.")
        } else {
            stopTraining()
            startTraining(motionType: newType)
        }
    }
    
    func startRecognition() {
        let motionInput = MotionInput { [weak self] series in
            if let motionType = self?.classifier.predict(x: magnitude(series3D: series)) {
                self?.onRecognition(motionType)
            }
        }
        
        do {
            try motionInput.start()
        } catch {
            print("Error when starting motion recognition:\n", error)
        }

        self.motionInput = motionInput
    }
    
    func stopRecognition() {
        stopMotionInput()
    }
    
    func startTraining(motionType: MotionType) {
        currentMotionType = motionType
        
        let motionInput = MotionInput { [weak self] series in
            self?.classifier.train(X: [magnitude(series3D: series)], y: [motionType])
        }
        
        do {
            try motionInput.start()
        } catch {
            print(error)
        }
        
        self.motionInput = motionInput
    }
    
    func stopTraining() {
        stopMotionInput()
    }
    
    private func stopMotionInput() {
        self.motionInput?.stop()
        self.motionInput = nil
    }
}
