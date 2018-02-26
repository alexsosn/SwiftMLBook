
import Foundation

struct AccuracyMetrics {
    
    //: ### Predictions quality validation.
    struct Metrics: CustomStringConvertible {
        let confusionMatrix: [[Int]]
        let normalizedConfusionMatrix: [[Double]]
        let accuracy: Double
        let precision: Double
        let recall: Double
        let f1Score: Double
        
        var description: String {
            return """
            Confusion Matrix:
            \(confusionMatrix)
            
            Normalized Confusion Matrix:
            \(normalizedConfusionMatrix)
            
            Accuracy: \(accuracy)
            Precision: \(precision)
            Recall: \(recall)
            F1-score: \(f1Score)
            """
        }
    }
    
    static func evaluateAccuracy(yVecTest: [Int], predictions: [Int]) -> Metrics {
        //: Calculating confusion matrix.
        let pairs: [(Int, Int)] = zip(predictions, yVecTest).map{ ($0.0, $0.1) }
        var confusionMatrix = [[0,0], [0,0]]
        for (p, t) in pairs {
            switch (p, t) {
            case (0, 0):
                confusionMatrix[0][0] += 1
                
            case (0, _):
                confusionMatrix[1][0] += 1
                
            case (_, 0):
                confusionMatrix[0][1] += 1
                
            case (_, _):
                confusionMatrix[1][1] += 1
            }
        }
        
        let totalCount = Double(yVecTest.count)
        // Normilize matrix by total count
        let normalizedConfusionMatrix = confusionMatrix.map{ $0.map{ Double($0)/totalCount } }
        
        //: Calculating accuracy.
        let truePredictionsCount = pairs.filter{ $0.0 == $0.1 }.count
        let accuracy = Double(truePredictionsCount) / totalCount
        
        //: Calculating precision, recall and F-score.
        let truePositive  = Double(pairs.filter{ $0.0 == $0.1 && $0.0 == 0 }.count)
        let falsePositive = Double(pairs.filter{ $0.0 != $0.1 && $0.0 == 0 }.count)
        let falseNegative = Double(pairs.filter{ $0.0 != $0.1 && $0.0 == 1 }.count)
        
        //: Precision.
        let precision = truePositive / (truePositive + falsePositive)
        //: Recall.
        let recall = truePositive / (truePositive + falseNegative)
        //: F1.
        let f1Score = 2 * precision * recall / (precision + recall)
        
        return Metrics(confusionMatrix: confusionMatrix, normalizedConfusionMatrix: normalizedConfusionMatrix, accuracy: accuracy, precision: precision, recall: recall, f1Score: f1Score)
    }
}
