//
//  SoftmaxLayer.swift
//  Pine
//
//  Created by Oleksandr on 4/23/17.
//  Copyright © 2017 OWL. All rights reserved.
//

import Foundation

// This is a classifier, with N discrete classes from 0 to N-1
// it gets a stream of N incoming numbers and computes the softmax
// function (exponentiate and normalize to sum to 1 as probabilities should)

public struct SoftmaxLayerOpt: LayerInOptProtocol, ClassificationLayerOptProtocol {
    public var numClasses: Int
    
    // This properties will be set by the Net class.
    public var inSx: Int = 0
    public var inSy: Int = 0
    public var inDepth: Int = 0
    
    public init (numClasses: Int) {
        self.numClasses = numClasses
    }
}

public class SoftmaxLayer: LossLayer {
    
    public var numInputs: Int
    public var outDepth: Int
    public var outSx: Int
    public var outSy: Int
    public var inAct: Volume?
    public var outAct: Volume?
    
    public var es: [Float] = []
    
    public init(opt: SoftmaxLayerOpt) {
        // computed
        numInputs = opt.numInputs()
        outDepth = numInputs
        outSx = 1
        outSy = 1
    }
    
    public func forward(_ V: inout Volume, isTraining: Bool) -> Volume {
        inAct = V
        
        let A = Volume(sx: 1, sy: 1, depth: outDepth, c: 0.0)
        
        // compute max activation
        var a_s = V.w
        let amax = V.w.max()!
        
        // compute exponentials (carefully to not blow up)
        var es = ArrayUtils.zerosFloat(outDepth)
        var esum = Float(0.0)
        for i in 0 ..< outDepth {
            
            let e = exp(a_s[i] - amax)
            esum += e
            es[i] = e
        }
        
        // normalize and output to sum to one
        for i in 0 ..< outDepth {
            
            es[i] /= esum
            A.w[i] = es[i]
        }
        
        self.es = es // save these for backprop
        outAct = A
        return outAct!
    }
    
    public func backward(_ y: Int) -> Float {
        
        // compute and accumulate gradient wrt weights and bias of this layer
        guard let x = inAct else {
            fatalError("inAct is nil")
        }
        x.dw = ArrayUtils.zerosFloat(x.w.count) // zero out the gradient of input Volume
        
        for i in 0 ..< outDepth {
            
            let indicator: Float = i == y ? 1.0 : 0.0
            let mul = -(indicator - es[i])
            x.dw[i] = mul
        }
        // loss is the class negative log likelihood
        return -log(es[y])
    }
    
    public func getParamsAndGrads() -> [ParamsAndGrads] {
        return []
    }
    
    public func assignParamsAndGrads(_ paramsAndGrads: [ParamsAndGrads]) ->() {
        
    }
}
