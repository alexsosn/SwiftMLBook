//
//  ReLULayer.swift
//  Pine
//
//  Created by Oleksandr on 4/23/17.
//  Copyright © 2017 OWL. All rights reserved.
//

import Foundation

// Implements ReLU nonlinearity elementwise
// x -> max(0, x)
// the output is in [0, inf)

public struct ReluLayerOpt: LayerInOptProtocol {
    public var inSx: Int = 0
    public var inSy: Int = 0
    public var inDepth: Int = 0
}

public class ReluLayer: HiddenLayer {
    public var outSx: Int
    public var outSy: Int
    public var outDepth: Int
    public var inAct: Volume?
    public var outAct: Volume?
    
    public init(opt: ReluLayerOpt) {
        
        // computed
        outSx = opt.inSx
        outSy = opt.inSy
        outDepth = opt.inDepth
    }
    
    public func forward(_ V: inout Volume, isTraining: Bool) -> Volume {
        inAct = V
        let V2 = V.clone()
        let N = V.w.count
        var V2w = V2.w
        for i in 0 ..< N {
            
            if V2w[i] < 0 { V2w[i] = 0 } // threshold at 0
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
            
            if V2.w[i] <= 0 {
                V.dw[i] = 0 // threshold
            } else {
                V.dw[i] = V2.dw[i]
            }
        }
    }
    
    public func getParamsAndGrads() -> [ParamsAndGrads] {
        return []
    }
    
    public func assignParamsAndGrads(_ paramsAndGrads: [ParamsAndGrads]) ->() {
        
    }
}
