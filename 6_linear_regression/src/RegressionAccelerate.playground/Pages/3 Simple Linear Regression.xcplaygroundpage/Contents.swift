//: [Previous](@previous)

import Foundation

let xVec: [Double] = [2,3,4,5]
let yVec: [Double] = [10,20,30,40]

let regression = SimpleLinearRegression()
regression.normalization = true
regression.train(xVec: xVec, yVec: yVec, learningRate: 0.1, maxSteps: 31)

regression.slope
regression.intercept

regression.xMean
regression.xStd

regression.yMean
regression.yStd

regression.predict(x: 7)
regression.cost(trueVec: yVec, predictedVec: regression.predict(xVec: xVec))


//: [Next](@next)
