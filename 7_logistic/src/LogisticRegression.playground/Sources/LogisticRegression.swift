//
//  LogisticRegression.swift
//  Pine
//
//  Created by Oleksandr on 5/4/17.
//  Copyright © 2017 OWL. All rights reserved.
//

import Foundation
import Accelerate


public class LogisticRegression {
    public var weights: [Double]!
    
    public init() {}
    
    public var normalization = false
    public var xMeanVec = [Double]()
    public var xStdVec = [Double]()
    
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
            
            // Save means and std-s for prediction phase.
            self.xMeanVec = xMeanVec
            self.xStdVec = xStdVec
            
            // Add first column of ones to matrix
            let designMatrix = prependColumnOfOnes(matrix: normalizedXMat, rows: sampleCount, columns: featureCount)
            
            gradientDescent(xMatFlattened: designMatrix, yVec: yVec, α: learningRate, maxSteps: maxSteps)
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
            
            return h
        } else {
            return hypothesis(xVec: [1.0]+xVec)
        }
    }
    
    private func hypothesis(xVec: [Double]) -> Double {
        var result = 0.0
        vDSP_dotprD(xVec, 1, weights, 1, &result, vDSP_Length(xVec.count))
        return 1.0 / (1.0 + exp(-result))
    }
    
    public func predict(xMat: [[Double]]) -> [Double] {
        let rows = xMat.count
        precondition(rows > 0)
        let columns = xMat.first!.count
        precondition(columns > 0)
        
        if normalization {
            let flattenedNormalizedX = xMat.map{
                return vecDivide(vecSubtract($0, xMeanVec), xStdVec)
                }.reduce([], +)
            
            // Add a column of ones in front of the matrix.
            let basisExpanded = prependColumnOfOnes(matrix: flattenedNormalizedX, rows: rows, columns: columns)
            let hVec = hypothesis(xMatFlattened: basisExpanded)
            
            return hVec
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
        var result = gemm(aMat: xMatFlattened, bMat: weights, rowsAC: sampleCount, colsBC: labelSize, colsA_rowsB: featureCount)
        
        // -h
        vDSP_vnegD(result, 1, &result, 1, vDSP_Length(sampleCount))
        
        // exp(-h)
        // vForce function for double-precision exponent.
        var outputLength = Int32(sampleCount)
        vvexp(&result, result, &outputLength)
        
        // 1.0 + exp(-h)
        var one = 1.0
        vDSP_vsaddD(result, 1, &one, &result, 1, vDSP_Length(sampleCount))
        
        // 1.0 / (1.0 + exp(-h))
        vDSP_svdivD(&one, result, 1, &result, 1, vDSP_Length(sampleCount))
        
        return result
    }
    
    // Cost function.
    // cost(y, h) = -sum(y.*log(h)+(1-y).*log(1-h))/m
    public func cost(trueVec: [Double], predictedVec: [Double]) -> Double {
        let count = trueVec.count
        // Calculate squared Euclidean distance.
        var result = 0.0
        
        var left = [Double](repeating: 0.0, count: count)
        var right = [Double](repeating: 0.0, count: count)
        
        // log(h)
        var outputLength = Int32(count)
        vvlog(&left, predictedVec, &outputLength)
        
        // -y.*log(h)
        left = vecMultiply(trueVec, left)
        
        // 1-y
        var minusOne = -1.0
        var oneMinusTrueVec = [Double](repeating: 0.0, count: count)
        vDSP_vsaddD(trueVec, 1, &minusOne, &oneMinusTrueVec, 1, vDSP_Length(count))
        vDSP_vnegD(oneMinusTrueVec, 1, &oneMinusTrueVec, 1, vDSP_Length(count))
        
        // 1-h
        var oneMinusPredictedVec = [Double](repeating: 0.0, count: count)
        vDSP_vsaddD(predictedVec, 1, &minusOne, &oneMinusPredictedVec, 1, vDSP_Length(count))
        vDSP_vnegD(oneMinusPredictedVec, 1, &oneMinusPredictedVec, 1, vDSP_Length(count))
        
        // log(1-h)
        vvlog(&right, oneMinusPredictedVec, &outputLength)
        
        // (1-y).*log(1-h)
        right = vecMultiply(oneMinusTrueVec, right)
        
        // left+right
        let sum = vecAdd(left, right)
        
        // sum()
        vDSP_sveD(sum, 1, &result, vDSP_Length(count))
        
        // Normalize by vector length.
        result/=(Double(count))
        return -result
    }
    
    // Derivative of a cost function.
    // x'*sum(h-y)
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


