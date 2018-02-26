
import Foundation
import GameplayKit

class Flow {
    func runWithCSV() {
        
        // Load data
        let (xMat, yVec) = DataProvider.loadCSVData()
        
        // Scikit-learn Decision Tree
        let sklDecisionTree = DecisionTree()
        
        let xSKLDecisionTree = xMat.map { (x: [Double]) -> DecisionTreeInput in
            return DecisionTreeInput(
                length: x[0],
                fluffy: x[1],
                color_light_black: x[2],
                color_pink_gold: x[3],
                color_purple_polka_dot: x[4],
                color_space_gray: x[5])
        }
        
        let predictionsSKLTree = try! xSKLDecisionTree
            .map(sklDecisionTree.prediction)
            .map{ prediction in
                return prediction.label == "rabbosaurus" ? 0 : 1
        }
        
        let groundTruth = yVec.map{ $0 == "rabbosaurus" ? 0 : 1 }
        
        let metricsSKLDecisionTree = AccuracyMetrics.evaluateAccuracy(yVecTest: groundTruth, predictions: predictionsSKLTree)
        print("")
        print("### Scikit-learn Decision Tree on data from csv file ###")
        print(metricsSKLDecisionTree)
        
        // Scikit-learn Random Forest
        let sklRandomForest = RandomForest()
        
        let xSKLRandomForest = xMat.map { (x: [Double]) -> RandomForestInput in
            return RandomForestInput(length: x[0],
                                     fluffy: x[1],
                                     color_light_black: x[2],
                                     color_pink_gold: x[3],
                                     color_purple_polka_dot: x[4],
                                     color_space_gray: x[5])
        }
        
        let predictionsSKLRandomForest = try! xSKLRandomForest.map(sklRandomForest.prediction).map{$0.label == "rabbosaurus" ? 0 : 1}
        
        let metricsSKLRandomForest = AccuracyMetrics.evaluateAccuracy(yVecTest: groundTruth, predictions: predictionsSKLRandomForest)
        print("")
        print("### Scikit-learn Random Forest on data from csv file ###")
        print(metricsSKLRandomForest)
    }
    
    func run(_ size: Int) {
        // GameplayKit Decision Tree
        print("")
        print("### GameplayKit Decision Tree on generated data ###")

        let (xMat, yVec) = DataProvider.generateData(size)
        let (xMatTrain, yVecTrain, xMatTest, yVecTest) = DataSplitter.split(xMat: xMat, yVec: yVec)
        let gkDecisionTree = GKDecisionTreeWrapper.train(xMatTrain: xMatTrain, yVecTrain: yVecTrain)
        
        //: Tree structure.
        let treeStructure = gkDecisionTree.description
        print("Tree structure:")
        print(treeStructure)
        
        let predictionsGKTree = GKDecisionTreeWrapper.makePredictions(xMatTest: xMatTest)
        let metricsGKTree = AccuracyMetrics.evaluateAccuracy(yVecTest: yVecTest, predictions: predictionsGKTree)
        print(metricsGKTree)
        
        // Scikit-learn Decision Tree
        
        let sklDecisionTree = DecisionTree()
        
        let xSKLDecisionTree = xMatTest.map { (x: [Int]) -> DecisionTreeInput in
            return DecisionTreeInput(length: Double(x[0]),
                                     fluffy: Double(x[1]),
                                     color_light_black: Double(x[2]),
                                     color_pink_gold: Double(x[3]),
                                     color_purple_polka_dot: Double(x[4]),
                                     color_space_gray: Double(x[5]))
        }
        
        let predictionsSKLTree = try! xSKLDecisionTree
            .map(sklDecisionTree.prediction)
            .map{ prediction in
                return prediction.label == "rabbosaurus" ? 0 : 1
        }
        
        let metricsSKLDecisionTree = AccuracyMetrics.evaluateAccuracy(yVecTest: yVecTest, predictions: predictionsSKLTree)
        print("")
        print("### Scikit-learn Decision Tree on generated data ###")
        print(metricsSKLDecisionTree)
        
        // Scikit-learn Random Forest
        let sklRandomForest = RandomForest()
        
        let xSKLRandomForest = xMatTest.map { (x: [Int]) -> RandomForestInput in
            return RandomForestInput(length: Double(x[0]),
                                     fluffy: Double(x[1]),
                                     color_light_black: Double(x[2]),
                                     color_pink_gold: Double(x[3]),
                                     color_purple_polka_dot: Double(x[4]),
                                     color_space_gray: Double(x[5]))
        }
        
        let predictionsSKLRandomForest = try! xSKLRandomForest.map(sklRandomForest.prediction).map{$0.label == "rabbosaurus" ? 0 : 1}
        
        let metricsSKLRandomForest = AccuracyMetrics.evaluateAccuracy(yVecTest: yVecTest, predictions: predictionsSKLRandomForest)
        print("")
        print("### Scikit-learn Random Forest on generated data ###")
        print(metricsSKLRandomForest)
    }
}
