
import Foundation
import GameplayKit

struct DataProvider {
    static func loadCSVData() -> ([[Double]], [String]) {
        guard let path = Bundle.main.path(forResource: "data_test", ofType: "csv"),
            let reader = StreamReader(path: path) else {
                fatalError()
        }
        
        var xMat = [[Double]]()
        var yVec = [String]()
        
        for (i, string) in reader.enumerated() where i > 0 {
            let elements = string.split(separator: "\t").map(String.init)
            
            assert(elements.count == 8, "Wrong number of features.")
            
            let length = Double(elements[1])!
            let fluffy = Double(elements[2] == "True")
            let color = elements[3...6].flatMap(Double.init)
            let featureVector = [length, fluffy] + color
            
            assert(featureVector.count == 6, "Wrong number of features.")
            
            xMat.append(featureVector)
            yVec.append(elements[7])
        }
        
        return (xMat, yVec)
    }
    
    static func generateData(_ size: Int) -> ([[Int]], [Int]) {
        //: ### Generating random dataset.
        //: Creating the source of pseudo-random numbers.
        let source = GKRandomSource()
        //: Creating random distributions to draw data from.
        let rabboLenDist = GKGaussianDistribution(randomSource: source, mean: 30, deviation: 5)
        let platyhogLenDist = GKGaussianDistribution(randomSource: source, mean: 20, deviation: 5)
        
        let rabboColorDist = GKRandomDistribution(lowestValue: 0, highestValue: 2)
        let platyhogColorDist = GKRandomDistribution(lowestValue: 1, highestValue: 3)
        //: Generating labels.
        let y: [Label] = (0..<size).map { _ in arc4random_uniform(2) > 0 ? .rabbosaurus : .platyhog }
        //: Convert labels to numbers.
        let labelVec = y.map{ $0.rawValue }
        //: Generating features.
        let featureMat: [[Int]] = y.map { label -> [Int] in
            if label == .rabbosaurus {
                let length = rabboLenDist.nextInt()
                let fluffy = Int(arc4random_uniform(100) < 90)
                
                let color = ColorFeature(rawValue: rabboColorDist.nextInt())!
                let colorOneHotEncoded = color.toOneHot()
                
                return [length, fluffy] + colorOneHotEncoded
            } else {
                let length = platyhogLenDist.nextInt()
                let fluffy = Int(arc4random_uniform(100) > 70)
                
                let color = ColorFeature(rawValue: platyhogColorDist.nextInt())!
                let colorOneHotEncoded = color.toOneHot()
                
                return [length, fluffy] + colorOneHotEncoded
            }
        }
        
        return (featureMat, labelVec)
    }
    
}
