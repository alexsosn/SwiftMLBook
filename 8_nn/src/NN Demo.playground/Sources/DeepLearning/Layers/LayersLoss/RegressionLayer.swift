//
//  RegressionLayer.swift
//  Pine
//
//  Created by Oleksandr on 4/23/17.
//  Copyright © 2017 OWL. All rights reserved.
//

import Foundation

// implements an L2 regression cost layer,
// so penalizes \sum_i(||x_i - y_i||^2), where x is its input
// and y is the user-provided array of "correct" values.

public struct RegressionLayerOpt: LayerInOptProtocol {
    public var numNeurons: Int
    
    // This properties will be set by the Net class.
    public var inSx: Int = 0
    public var inSy: Int = 0
    public var inDepth: Int = 0
    
    public init(numNeurons: Int = 1) {
        self.numNeurons = numNeurons
    }
}

public class RegressionLayer: LossLayer {
    public var numInputs: Int
    public var outDepth: Int
    public var outSx: Int
    public var outSy: Int
    public var inAct: Volume?
    public var outAct: Volume?
    
    public init(opt: RegressionLayerOpt) {
        
        // computed
        numInputs = opt.numInputs()
        outDepth = numInputs
        outSx = 1
        outSy = 1
    }
    
    public func forward(_ V: inout Volume, isTraining: Bool) -> Volume {
        inAct = V
        outAct = V
        return V // identity function
    }
    
    // y is a list here of size numInputs
    // or it can be a number if only one value is regressed
    // or it can be a struct {dim: i, val: x} where we only want to
    // regress on dimension i and asking it to have value x
    
    public func backward(_ y: [Float]) -> Float {
        
        // compute and accumulate gradient wrt weights and bias of this layer
        guard let x = inAct else {
            fatalError("inAct is nil")
        }
        x.dw = ArrayUtils.zerosFloat(x.w.count) // zero out the gradient of input Volume
        var loss = Float(0.0)
        for i in 0 ..< outDepth {
            let dy = x.w[i] - y[i]
            x.dw[i] = dy
            loss += 0.5*dy*dy
        }
        inAct = x
        return loss
    }
    
    public func backward(_ y: Float) -> Float {
        // compute and accumulate gradient wrt weights and bias of this layer
        guard let x = inAct else {
            fatalError("inAct is nil")
        }
        x.dw = ArrayUtils.zerosFloat(x.w.count) // zero out the gradient of input Volume
        var loss = Float(0.0)
        // lets hope that only one number is being regressed
        let dy = x.w[0] - y
        x.dw[0] = dy
        loss += 0.5*dy*dy
        return loss
    }
    
    public func backward(_ y: Int) -> Float {
        return backward(Float(y))
    }
    
    public func backward(_ y: (dim: Int, val: Float)) -> Float {
        // compute and accumulate gradient wrt weights and bias of this layer
        guard let x = inAct else {
            fatalError("inAct is nil")
        }
        x.dw = ArrayUtils.zerosFloat(x.w.count) // zero out the gradient of input Volume
        var loss = Float(0.0)
        // assume it is a struct with entries .dim and .val
        // and we pass gradient only along dimension dim to be equal to val
        let i = y.dim
        let yi = y.val
        let dy = x.w[i] - yi
        x.dw[i] = dy
        loss += 0.5*dy*dy
        return loss
    }
    
    public func getParamsAndGrads() -> [ParamsAndGrads] {
        return []
    }
    
    public func assignParamsAndGrads(_ paramsAndGrads: [ParamsAndGrads]) ->() {
        
    }
}
