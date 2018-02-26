//: Some toy data
let xMat: [[Double]] = [
    [1,6],
    [2,6],
    [3,5],
    [4,8],
    [15, 1],
    [12, 2],
    [10, 1]
]

let yVec: [Double] = [0,0,0,0,1,1,1]
let testXMat: [[Double]] = [[2, 5], [6, 1], [3, 7], [13, 3]]
let test_y: [Double] = [0, 1, 0, 1]

//: Training logistic regression
let regression = LogisticRegression()
regression.train(xMat: xMat, yVec: yVec, learningRate: 0.1, maxSteps: 10000)

//: Predicting things
let pred_y = regression.predict(xMat: testXMat)
print("Pridiction: ", pred_y, "true: ", test_y)
print("Weights: ", regression.weights)
print("Difference: ", Euclidean.distance(pred_y, test_y))
let ε = 0.1

Euclidean.distance(pred_y, test_y) < ε
let cost = regression.cost(trueVec: yVec, predictedVec: regression.predict(xMat: xMat))
print("Cost: ", cost)

//: Logistic regression with data normalization.
let normalizedRegression = LogisticRegression()
normalizedRegression.normalization = true
normalizedRegression.train(xMat: xMat, yVec: yVec, learningRate: 0.1, maxSteps: 10000)

let pred_y_norm = normalizedRegression.predict(xMat: testXMat)
print("Pridiction: ", pred_y, "true: ", test_y)
print("Weights: ", normalizedRegression.weights)
print("Difference: ", Euclidean.distance(pred_y, test_y))

Euclidean.distance(pred_y, test_y) < ε
let cost_norm = regression.cost(trueVec: yVec, predictedVec: regression.predict(xMat: xMat))
print("Cost: ", cost)
cost < ε

