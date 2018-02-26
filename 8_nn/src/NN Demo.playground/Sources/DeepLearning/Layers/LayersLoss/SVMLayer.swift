//
//  SVMLayer.swift
//  Pine
//
//  Created by Oleksandr on 4/23/17.
//  Copyright © 2017 OWL. All rights reserved.
//

import Foundation

public struct SVMLayerOpt: LayerInOptProtocol, ClassificationLayerOptProtocol {
    public var numClasses: Int
    
    // This properties will be set by the Net class.
    public var inSx: Int = 0
    public var inSy: Int = 0
    public var inDepth: Int = 0
    
    public init (numClasses: Int) {
        self.numClasses = numClasses
    }
}

public class SVMLayer: LossLayer {
    public var numInputs: Int
    public var outDepth: Int
    public var outSx: Int
    public var outSy: Int
    public var inAct: Volume?
    public var outAct: Volume?
    
    public init(opt: SVMLayerOpt){
        // computed
        numInputs = opt.numInputs()
        outDepth = numInputs
        outSx = 1
        outSy = 1
    }
    
    public func forward(_ V: inout Volume, isTraining: Bool) -> Volume {
        inAct = V
        outAct = V // nothing to do, output raw scores
        return V
    }
    
    public func backward(_ y: Int) -> Float {
        
        // compute and accumulate gradient wrt weights and bias of this layer
        guard let x = inAct else {
            fatalError("inAct is nil")
        }
        
        x.dw = ArrayUtils.zerosFloat(x.w.count) // zero out the gradient of input Volume
        
        // we're using structured loss here, which means that the score
        // of the ground truth should be higher than the score of any other
        // class, by a margin
        let yscore = x.w[y] // score of ground truth
        let margin = Float(1.0)
        var loss = Float(0.0)
        for i in 0 ..< outDepth {
            
            if y == i { continue }
            let ydiff = -yscore + x.w[i] + margin
            if ydiff > 0 {
                // violating dimension, apply loss
                x.dw[i] += 1
                x.dw[y] -= 1
                loss += ydiff
            }
        }
        inAct = x
        return loss
    }
    
    public func getParamsAndGrads() -> [ParamsAndGrads] {
        return []
    }
    
    public func assignParamsAndGrads(_ paramsAndGrads: [ParamsAndGrads]) ->() {
        
    }
}

