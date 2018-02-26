
import Foundation
import GameplayKit

struct GKDecisionTreeWrapper {
    static var tree: GKDecisionTree?
    
    static func train(xMatTrain: [[Int]], yVecTrain: [Int]) -> GKDecisionTree {
        //: ### Creating and training the model (decision tree).
        let decisionTree = GKDecisionTree(
            examples: xMatTrain as [[NSNumber]],
            actions: yVecTrain as [NSNumber],
            attributes: FeatureLabels.featureLabels as [NSString]
        )
        tree = decisionTree
        return decisionTree
    }
    
    static func makePredictions() -> (Label?, Label?) {
        //: ### Making predictions.
        // First case.
        let whoIsThis1: [String: Int] = [
            "Length": 26,
            "Fluffy": 0, // false
            "IsSpaceGray": 0,
            "IsLightBlack": 1, // true
            "IsPinkGold": 0,
            "IsPurplePolkaDot": 0
        ]
        
        let predicted1 = tree?.findAction(forAnswers: whoIsThis1 as [AnyHashable: NSNumber])
        var result1: Label? = nil
        if let predicted1 = predicted1 as? Int {
            result1 = Label(rawValue: predicted1)
        }
        
        
        // Second case.
        let whoIsThis2: [String: Int] = [
            "Length": 24,
            "Fluffy": 1,
            "IsSpaceGray": 0,
            "IsLightBlack": 1,
            "IsPinkGold": 0,
            "IsPurplePolkaDot": 0
        ]
        
        let predicted2 = tree?.findAction(forAnswers: whoIsThis2 as [AnyHashable : NSObjectProtocol])
        var result2: Label? = nil
        if let predicted2 = predicted2 as? Int {
            result2 = Label(rawValue: predicted2)
        }
        return (result1, result2)
    }
    
    static func makePredictions(xMatTest: [[Int]]) -> [Int] {
        // Calculating predictions for all test cases.
        // Converting feature matrix to an array of dictionaries.
        
        let dicts: [[AnyHashable: Int]] = xMatTest.map { value -> [AnyHashable: Int] in
            return Dictionary(uniqueKeysWithValues: zip(FeatureLabels.featureLabels as [AnyHashable], value))
        }
        
        let predictions: [Int] = dicts.map{ sample in
            guard let result = tree?.findAction(forAnswers: sample as [AnyHashable: NSObjectProtocol]) else {
                return -1
            }
            return result as! Int
        }
        return predictions
    }
}
