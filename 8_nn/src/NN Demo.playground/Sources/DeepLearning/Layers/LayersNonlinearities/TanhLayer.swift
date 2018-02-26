//
//  TanhLayer.swift
//  Pine
//
//  Created by Oleksandr on 4/23/17.
//  Copyright © 2017 OWL. All rights reserved.
//

import Foundation

// Implements Tanh nnonlinearity elementwise
// x -> tanh(x)
// so the output is between -1 and 1.

public struct TanhLayerOpt: LayerInOptProtocol {
    public var inSx: Int = 0
    public var inSy: Int = 0
    public var inDepth: Int = 0
}

public class TanhLayer: HiddenLayer {
    public var outSx: Int
    public var outSy: Int
    public var outDepth: Int
    public var inAct: Volume?
    public var outAct: Volume?
    
    public init(opt: TanhLayerOpt) {
        
        // computed
        outSx = opt.inSx
        outSy = opt.inSy
        outDepth = opt.inDepth
    }
    
    public func forward(_ V: inout Volume, isTraining: Bool) -> Volume {
        inAct = V
        let V2 = V.cloneAndZero()
        let N = V.w.count
        for i in 0 ..< N {
            
            V2.w[i] = tanh(V.w[i])
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
            V.dw[i] = (1.0 - v2wi * v2wi) * V2.dw[i]
        }
    }
    
    public func getParamsAndGrads() -> [ParamsAndGrads] {
        return []
    }
    
    public func assignParamsAndGrads(_ paramsAndGrads: [ParamsAndGrads]) ->() {
        
    }
}
