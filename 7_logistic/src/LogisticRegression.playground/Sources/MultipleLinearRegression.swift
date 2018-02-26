//
//  MultipleLinearRegression.swift
//  Regression
//
//  Created by Alex Sosnovshchenko on 1/9/17.
//  Copyright © 2017 Alex Sosnovshchenko. All rights reserved.
//

import Foundation
import Accelerate

enum MultipleLinearRegressionError: Error {
    case noUniqueSolution
}

public class MultipleLinearRegression {
    public var weights: [Double]!
    
    public init() {}
    
    public var normalization = false
    public var xMeanVec = [Double]()
    public var xStdVec = [Double]()
    public var yMean = 0.0
    public var yStd = 0.0
    
    //: $\mathbf{A}^\top\mathbf{A}\mathbf{c}=\mathbf{A}^\top\mathbf{y}$
    
    public func solve(xMat: [[Double]], yVec: [Double]) throws {
        let sampleCount = xMat.count
        precondition(sampleCount == yVec.count, "The number of samples in xMat should be equal to the number of labels in yVec.")
        precondition(sampleCount > 0, "xMat should contain at least one sample.")
        precondition(xMat.first!.count > 0, "Samples should have at least one feature.")
        let featureCount = xMat.first!.count + 1
        precondition(featureCount < sampleCount)
        
        weights = [Double](repeating: 1.0, count: featureCount)
        // Flatten and prepend a column of ones.
        let flattened = xMat.map{[1.0]+$0}.reduce([], +)
        let squareMat = gemm(aMat: flattened, bMat: flattened, rowsAC: featureCount, colsBC: featureCount, colsA_rowsB: sampleCount, transposeA: true)
        let invertedOptional = try? inverse(squareMat, dimension: featureCount)
        guard let inverted = invertedOptional else {
            throw MultipleLinearRegressionError.noUniqueSolution
        }
        let inverseXTranspose = gemm(aMat: inverted, bMat: flattened, rowsAC: featureCount, colsBC: sampleCount, colsA_rowsB: featureCount, transposeB: true)
        let labelSize = 1
        let result = gemm(aMat: inverseXTranspose, bMat: yVec, rowsAC: featureCount, colsBC: labelSize, colsA_rowsB: sampleCount)
        weights = result
        
        /*
         //  Solve the matrix for the parameters Θ (DGELS)
         let jobTChar = "N" as NSString
         var jobT : Int8 = Int8(jobTChar.character(at: 0))          //  not transposed
         var m : __CLPK_integer = __CLPK_integer(trainData.size)
         var n : __CLPK_integer = __CLPK_integer(numColumns)
         var nrhs = __CLPK_integer(outputDimension)
         var work : [Double] = [0.0]
         var lwork : __CLPK_integer = -1        //  Ask for the best size of the work array
         var info : __CLPK_integer = 0
         dgels_(&jobT, &m, &n, &nrhs, &A, &m, &y, &m, &work, &lwork, &info)
         if (info != 0 || work[0] < 1) {
         throw LinearRegressionError.matrixSolutionError
         }
         lwork = __CLPK_integer(work[0])
         work = [Double](repeating: 0.0, count: Int(work[0]))
         dgels_(&jobT, &m, &n, &nrhs, &A, &m, &y, &m, &work, &lwork, &info)
         if (info != 0 || work[0] < 1) {
         throw LinearRegressionError.matrixSolutionError
         }
         
         //  Extract the parameters from the results
         for output in 0..<outputDimension {
         for parameter in 0..<numColumns {
         Θ[output].append(y[output * trainData.size + parameter])
         }
         }
         */
    }
    
    // weights = (X'X)^-1*X'y
    public func solveNaïve(xMat: [[Double]], yVec: [Double]) throws {
        let sampleCount = xMat.count
        precondition(sampleCount == yVec.count, "The number of samples in xMat should be equal to the number of labels in yVec.")
        precondition(sampleCount > 0, "xMat should contain at least one sample.")
        precondition(xMat.first!.count > 0, "Samples should have at least one feature.")
        let featureCount = xMat.first!.count + 1
        precondition(featureCount < sampleCount)
        
        weights = [Double](repeating: 1.0, count: featureCount)
        // Flatten and prepend a column of ones.
        let flattened = xMat.map{[1.0]+$0}.reduce([], +)
        let squareMat = gemm(aMat: flattened, bMat: flattened, rowsAC: featureCount, colsBC: featureCount, colsA_rowsB: sampleCount, transposeA: true)
        let invertedOptional = try? inverse(squareMat, dimension: featureCount)
        guard let inverted = invertedOptional else {
            throw MultipleLinearRegressionError.noUniqueSolution
        }
        let inverseXTranspose = gemm(aMat: inverted, bMat: flattened, rowsAC: featureCount, colsBC: sampleCount, colsA_rowsB: featureCount, transposeB: true)
        let labelSize = 1
        let result = gemm(aMat: inverseXTranspose, bMat: yVec, rowsAC: featureCount, colsBC: labelSize, colsA_rowsB: sampleCount)
        weights = result
    }
    
    private func prependColumnOfOnes(matrix: [Double], rows: Int, columns: Int) -> [Double] {
        let weightsCount = columns+1
        
        var withFirstDummyColumn = [Double](repeating: 1.0, count: rows * (columns+1))
        for row in 0..<rows {
            for column in 1..<weightsCount {
                withFirstDummyColumn[row*weightsCount + column] = matrix[row*columns + column-1]
            }
        }
        return withFirstDummyColumn
    }
    
    public func train(xMat: [[Double]], yVec: [Double], learningRate: Double, maxSteps: Int) {
        precondition(maxSteps > 0, "The number of learning iterations should be grater then 0.")
        let sampleCount = xMat.count
        precondition(sampleCount == yVec.count, "The number of samples in xMat should be equal to the number of labels in yVec.")
        precondition(sampleCount > 0, "xMat should contain at least one sample.")
        precondition(xMat.first!.count > 0, "Samples should have at least one feature.")
        let featureCount = xMat.first!.count
        let weightsCount = featureCount+1
        
        weights = [Double](repeating: 1.0, count: weightsCount)
        if normalization {
            // Flatten
            let flattenedXMat = xMat.reduce([], +)
            let (normalizedXMat, xMeanVec, xStdVec) = matNormalize(matrix: flattenedXMat, rows: sampleCount, columns: featureCount)
            let (normalizedYVec, yMean, yStd) = vecNormalize(vec: yVec)
            
            // Save means and std-s for prediction phase.
            self.xMeanVec = xMeanVec
            self.xStdVec = xStdVec
            self.yMean = yMean
            self.yStd = yStd
            
            // Add first column of ones to matrix
            let designMatrix = prependColumnOfOnes(matrix: normalizedXMat, rows: sampleCount, columns: featureCount)
            
            gradientDescent(xMatFlattened: designMatrix, yVec: normalizedYVec, α: learningRate, maxSteps: maxSteps)
        } else {
            // Flatten and prepend a column of ones.
            let flattenedXMat = xMat.map{[1.0]+$0}.reduce([], +)
            gradientDescent(xMatFlattened: flattenedXMat, yVec: yVec, α: learningRate, maxSteps: maxSteps)
        }
    }
    
    public func predict(xVec: [Double]) -> Double {
        if normalization {
            let input = xVec
            let differenceVec = vecSubtract(input, xMeanVec)
            let normalizedInputVec = vecDivide(differenceVec, xStdVec)
            
            let h = hypothesis(xVec: [1.0]+normalizedInputVec)
            
            return h * yStd + yMean
        } else {
            return hypothesis(xVec: [1.0]+xVec)
        }
    }
    
    private func hypothesis(xVec: [Double]) -> Double {
        var result = 0.0
        vDSP_dotprD(xVec, 1, weights, 1, &result, vDSP_Length(xVec.count))
        return result
    }
    
    public func predict(xMat: [[Double]]) -> [Double] {
        let rows = xMat.count
        precondition(rows > 0)
        let columns = xMat.first!.count
        precondition(columns > 0)
        
        if normalization {
            let flattenedNormalizedX = xMat.map {
                return vecDivide(vecSubtract($0, xMeanVec), xStdVec)
                }.reduce([], +)
            
            // Add a column of ones in front of the matrix.
            let basisExpanded = prependColumnOfOnes(matrix: flattenedNormalizedX, rows: rows, columns: columns)
            
            let hVec = hypothesis(xMatFlattened: basisExpanded)
            let outputSize = hVec.count
            let productVec = vecMultiply(hVec, [Double](repeating: yStd, count: outputSize))
            let outputVec = vecAdd(productVec, [Double](repeating: yMean, count: outputSize))
            
            return outputVec
        } else {
            // Flatten and prepend a column of ones.
            let flattened = xMat.map{[1.0]+$0}.reduce([], +)
            return hypothesis(xMatFlattened: flattened)
        }
    }
    
    private func hypothesis(xMatFlattened: [Double]) -> [Double] {
        let matCount = xMatFlattened.count
        let featureCount = weights.count
        precondition(matCount > 0)
        let sampleCount = matCount/featureCount
        precondition(sampleCount*featureCount == matCount)
        let labelSize = 1
        let result = gemm(aMat: xMatFlattened, bMat: weights, rowsAC: sampleCount, colsBC: labelSize, colsA_rowsB: featureCount)
        return result
    }
    
    // Least squares cost function.
    // The goal of the training is to minimize cost function.
    public func cost(trueVec: [Double], predictedVec: [Double]) -> Double {
        let count = trueVec.count
        // Calculate squared Euclidean distance.
        var result = 0.0
        vDSP_distancesqD(trueVec, 1, predictedVec, 1, &result, 1)
        // Normalize by vector length.
        result/=(2*Double(count))
        return result
    }
    
    // derivative of a cost function
    private func costGradient(trueVec: [Double], predictedVec: [Double], xMatFlattened: [Double]) -> [Double] {
        let matCount = xMatFlattened.count
        let featureCount = weights.count
        precondition(matCount > 0)
        precondition(Double(matCount).truncatingRemainder(dividingBy: Double(featureCount)) == 0)
        let sampleCount = trueVec.count
        precondition(sampleCount > 0)
        precondition(sampleCount*featureCount == matCount)
        let labelSize = 1
        
        let diffVec = vecSubtract(predictedVec, trueVec)
        
        // Normalize by vector length.
        let scaleBy = 1/Double(sampleCount)
        let result = gemm(aMat: xMatFlattened, bMat: diffVec, rowsAC: featureCount, colsBC: labelSize, colsA_rowsB: sampleCount, transposeA: true, α: scaleBy)
        
        return result
    }
    
    // alpha is a learning rate
    private func gradientDescentStep(xMatFlattened: [Double], yVec: [Double], α: Double) -> [Double] {
        
        // Calculate hypothesis predictions.
        let hVec = hypothesis(xMatFlattened: xMatFlattened)
        // Calculate gradient with respect to parameters.
        let gradient = costGradient(trueVec: yVec, predictedVec: hVec, xMatFlattened: xMatFlattened)
        let featureCount = gradient.count
        
        // newWeights = weights - α*gradient
        var alpha = α
        var scaledGradient = [Double](repeating: 0.0, count: featureCount)
        vDSP_vsmulD(gradient, 1, &alpha, &scaledGradient, 1, vDSP_Length(featureCount))
        
        let newWeights = vecSubtract(weights, scaledGradient)
        
        return newWeights
    }
    
    private func gradientDescent(xMatFlattened: [Double], yVec: [Double], α: Double, maxSteps: Int) {
        for _ in 0 ..< maxSteps {
            let newWeights = gradientDescentStep(xMatFlattened: xMatFlattened, yVec: yVec, α: α)
            if newWeights==weights {
                print("convergence")
                break
            } // convergence
            weights = newWeights
        }
    }
}



