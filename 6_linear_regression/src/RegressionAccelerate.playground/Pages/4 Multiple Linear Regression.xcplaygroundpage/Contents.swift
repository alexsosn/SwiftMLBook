
import Foundation

let testBundle = Bundle.main
let fileURL = testBundle.url(forResource: "regression_data", withExtension: "csv")!

guard let aStreamReader = StreamReader(path: fileURL.path, delimiter: "\r", encoding: .utf8, chunkSize: 4096 ) else { fatalError() }

var dates = [Date]()
var pages = [Int]()

let dateFormatter = DateFormatter()
dateFormatter.dateStyle = .short
dateFormatter.timeStyle = .none
dateFormatter.timeZone = TimeZone(secondsFromGMT: 2)

while let line = aStreamReader.nextLine() {
    let elements = line.components(separatedBy: ",")
    let dateString = elements[0]
    
    let date = dateFormatter.date(from: dateString)!
    dates.append(date)
    let pagesCount = Int(elements[1])!
    pages.append(pagesCount)
}
aStreamReader.close()

let x: [Double] = pages.map{Double($0)}
let y: [Double] = dates.map{$0.timeIntervalSince1970}

print(x)
print(y)

let simpleRegression = SimpleLinearRegression()
simpleRegression.normalization = true
simpleRegression.train(xVec: x, yVec: y, learningRate: 0.1, maxSteps: 1000)
simpleRegression.predict(x: 300)
simpleRegression.predict(xVec: [300.0])


let xMat: [[Double]] = x.map{[$0]}
var multipleRegression = MultipleLinearRegression()
multipleRegression.normalization = true
multipleRegression.train(xMat: xMat, yVec: y, learningRate: 0.1, maxSteps: 1000)
let result = multipleRegression.predict(xVec: [300.0])
Date(timeIntervalSince1970: result)
multipleRegression.predict(xMat: [[300.0]])

multipleRegression = MultipleLinearRegression()
try! multipleRegression.solveNaïve(xMat: xMat, yVec: y)
let result2 = multipleRegression.predict(xVec: [300.0])
multipleRegression.predict(xMat: [[300.0]])

Date(timeIntervalSince1970: result2)
