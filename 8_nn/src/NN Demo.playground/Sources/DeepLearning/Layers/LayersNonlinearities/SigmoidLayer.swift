//
//  SigmoidLayer.swift
//  Pine
//
//  Created by Oleksandr on 4/23/17.
//  Copyright © 2017 OWL. All rights reserved.
//

import Foundation

// Implements Sigmoid nnonlinearity elementwise
// x -> 1/(1+e^(-x))
// so the output is between 0 and 1.

public struct SigmoidLayerOpt: LayerInOptProtocol {
    public var inSx: Int = 0
    public var inSy: Int = 0
    public var inDepth: Int = 0
}

public class SigmoidLayer: HiddenLayer {
    public var outSx: Int
    public var outSy: Int
    public var outDepth: Int
    public var inAct: Volume?
    public var outAct: Volume?
    
    public init(opt: SigmoidLayerOpt){
        
        // computed
        outSx = opt.inSx
        outSy = opt.inSy
        outDepth = opt.inDepth
    }
    // http://memkite.com/blog/2014/12/15/data-parallel-programming-with-metal-and-swift-for-iphoneipad-gpu/
    public func forward(_ V: inout Volume, isTraining: Bool) -> Volume {
        inAct = V
        let V2 = V.cloneAndZero()
        let N = V.w.count
        var V2w = V2.w
        var Vw = V.w
        for i in 0 ..< N {
            
            V2w[i] = Float(1.0)/(1.0+exp(-Vw[i]))
        }
        outAct = V2
        return outAct!
    }
    
    public func backward() -> () {
        guard let V = inAct,
            let V2 = outAct
            else { // we need to set dw of this
                fatalError("inAct or outAct is nil")
        }
        let N = V.w.count
        V.dw = ArrayUtils.zerosFloat(N) // zero out gradient wrt data
        for i in 0 ..< N {
            
            let v2wi = V2.w[i]
            V.dw[i] =  v2wi * (1.0 - v2wi) * V2.dw[i]
        }
    }
    
    public func getParamsAndGrads() -> [ParamsAndGrads] {
        return []
    }
    
    public func assignParamsAndGrads(_ paramsAndGrads: [ParamsAndGrads]) ->() {
        
    }
}
