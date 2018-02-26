//
//  LinearRegression.swift
//  Regression
//
//  Created by Alex Sosnovshchenko on 1/9/17.
//  Copyright © 2017 Alex Sosnovshchenko. All rights reserved.
//

import Foundation
import Accelerate

public class SimpleLinearRegression {
    public var slope = 1.0
    public var intercept = 0.0
    
    // Normalization
    public var normalization = false
    public var xMean = 0.0
    public var xStd = 0.0
    public var yMean = 0.0
    public var yStd = 0.0
    
    public init() {}
    
    public func train(xVec: [Double], yVec: [Double], learningRate: Double, maxSteps: Int) {
        precondition(xVec.count == yVec.count)
        precondition(maxSteps > 0)
        if normalization {
            let (normalizedXVec, xMean, xStd) = vecNormalize(vec: xVec)
            let (normalizedYVec, yMean, yStd) = vecNormalize(vec: yVec)
            
            // Save means and std-s for prediction phase.
            self.xMean = xMean
            self.xStd = xStd
            self.yMean = yMean
            self.yStd = yStd
            
            gradientDescent(xVec: normalizedXVec, yVec: normalizedYVec, α: learningRate, maxSteps: maxSteps)
        } else {
            gradientDescent(xVec: xVec, yVec: yVec, α: learningRate, maxSteps: maxSteps)
        }
    }
    
    private func hypothesis(x: Double) -> Double {
        return slope*x + intercept
    }
    
    public func predict(x: Double) -> Double {
        if normalization {
            return hypothesis(x: (x-xMean)/xStd) * yStd + yMean
        } else {
            return hypothesis(x: x)
        }
    }
    
    private func hypothesis(xVec: [Double]) -> [Double] {
        let count = xVec.count
        var scaledVec = [Double](repeating: 0.0, count: count)
        vDSP_vsmulD(xVec, 1, &slope, &scaledVec, 1, vDSP_Length(count))
        var resultVec = [Double](repeating: 0.0, count: count)
        vDSP_vsaddD(scaledVec, 1, &intercept, &resultVec, 1, vDSP_Length(count))
        return resultVec
    }
    
    public func predict(xVec: [Double]) -> [Double] {
        if normalization {
            let count = xVec.count
            // Normalize
            var centeredVec = [Double](repeating: 0.0, count: count)
            var negMean = -xMean
            vDSP_vsaddD(xVec, 1, &negMean, &centeredVec, 1, vDSP_Length(count))
            
            var scaledVec = [Double](repeating: 0.0, count: count)
            vDSP_vsdivD(centeredVec, 1, &xStd, &scaledVec, 1, vDSP_Length(count))
            
            // Predict
            let hVec = hypothesis(xVec: scaledVec)
            
            // Denormalize
            var unScaledVec = [Double](repeating: 0.0, count: count)
            vDSP_vsmulD(hVec, 1, &yStd, &unScaledVec, 1, vDSP_Length(count))
            var resultVec = [Double](repeating: 0.0, count: count)
            vDSP_vsaddD(unScaledVec, 1, &yMean, &resultVec, 1, vDSP_Length(count))
            return resultVec
        } else {
            return hypothesis(xVec: xVec)
        }
    }
    
    // Least squares cost function.
    // The goal of the training is to minimize cost function.
    public func cost(trueVec: [Double], predictedVec: [Double]) -> Double {
        let count = trueVec.count
        // Calculate squared Euclidean distance.
        // https://en.wikipedia.org/wiki/Euclidean_distance
        var result = 0.0
        vDSP_distancesqD(trueVec, 1, predictedVec, 1, &result, 1)
        // Normalize by vector length.
        result/=(2*Double(count))
        return result
    }
    
    // derivative of a cost function
    private func costGradient(trueVec: [Double], predictedVec: [Double], xVec: [Double]) -> Double {
        let count = trueVec.count
        // gradient = (predicted-y)*x
        var diffVec = [Double](repeating: 0.0, count: count)
        // Note, that vDSP_vsubD takes operands in reverse order: f(b,a) = (a-b)
        vDSP_vsubD(trueVec, 1, predictedVec, 1, &diffVec, 1, vDSP_Length(count))
        
        var result = 0.0
        vDSP_dotprD(diffVec, 1, xVec, 1, &result, vDSP_Length(count))
        
        // Normalize by vector length.
        return result/Double(count)
    }
    
    // alpha is a learning rate
    private func gradientDescentStep(xVec: [Double], yVec: [Double], α: Double) -> (Double, Double) {
        // Calculate hypothesis predictions.
        let hVec = hypothesis(xVec: xVec)
        // Calculate gradient with respect to parameters.
        let slopeGradient = costGradient(trueVec: yVec, predictedVec: hVec, xVec: xVec)
        let newSlope = slope - α*slopeGradient
        
        let dummyVec = [Double](repeating: 1.0, count: xVec.count)
        let interceptGradient = costGradient(trueVec: yVec, predictedVec: hVec, xVec: dummyVec)
        let newIntercept = intercept - α*interceptGradient
        
        return (newSlope, newIntercept)
    }
    
    private func gradientDescent(xVec: [Double], yVec: [Double], α: Double, maxSteps: Int) {
        for _ in 0 ..< maxSteps {
            let (newSlope, newIntercept) = gradientDescentStep(xVec: xVec, yVec: yVec, α: α)
            if (newSlope==slope && newIntercept==intercept) { break } // convergence
            slope = newSlope
            intercept = newIntercept
        }
    }
}
