
import Foundation
import GameplayKit

struct DataSplitter {
    //: ### Splitting the data into training and test sets.
    static func split<T>(xMat: [[T]], yVec: [T], testSize: Double = 0.25)
        -> (xMatTrain: [[T]], yVecTrain: [T], xMatTest: [[T]], yVecTest: [T]) {
            let shuffler = GKShuffledDistribution(lowestValue: 0, highestValue: xMat.count - 1)
            
            var xMatShuffled: [[T]] = []
            var yVecShuffled: [T] = []
            
            for _ in 0..<xMat.count {
                let randIndex = shuffler.nextInt()
                xMatShuffled.append(xMat[randIndex])
                yVecShuffled.append(yVec[randIndex])
            }
            
            let split = Int(ceil(Double(xMat.count) * testSize))
            
            let xMatTrain = xMatShuffled[split + 1..<xMatShuffled.count]
            let yVecTrain = yVecShuffled[split + 1..<yVecShuffled.count]
            let xMatTest = xMatShuffled[0...split]
            let yVecTest = yVecShuffled[0...split]
            
            return (Array(xMatTrain), Array(yVecTrain), Array(xMatTest), Array(yVecTest))
    }
}
