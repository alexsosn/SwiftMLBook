//
//  MaxoutLayer.swift
//  Pine
//
//  Created by Oleksandr on 4/23/17.
//  Copyright © 2017 OWL. All rights reserved.
//

import Foundation

// Implements Maxout nnonlinearity that computes
// x -> max(x)
// where x is a vector of size groupSize. Ideally of course,
// the input size should be exactly divisible by groupSize

public struct MaxoutLayerOpt: LayerInOptProtocol {
    
    public var inSx: Int = 1
    public var inSy: Int = 1
    public var inDepth: Int = 1
    
    public var groupSize: Int
    
    public init (groupSize: Int?) {
        self.groupSize = groupSize ?? 2
    }
}

public class MaxoutLayer: HiddenLayer {
    public var groupSize: Int
    public var outSx: Int
    public var outSy: Int
    public var outDepth: Int
    public var inAct: Volume?
    public var outAct: Volume?
    public var switches: [Int]
    
    public init(opt: MaxoutLayerOpt){
        
        // required
        groupSize = opt.groupSize
        
        // computed
        outSx = opt.inSx
        outSy = opt.inSy
        outDepth = opt.inDepth / groupSize // WARNING: floor was here
        
        switches = ArrayUtils.zerosInt(outSx*outSy*outDepth) // useful for backprop
    }
    
    public func forward(_ V: inout Volume, isTraining: Bool) -> Volume {
        inAct = V
        let N = outDepth
        let V2 = Volume(sx: outSx, sy: outSy, depth: outDepth, c: 0.0)
        
        // optimization branch. If we're operating on 1D arrays we dont have
        // to worry about keeping track of x,y,d coordinates inside
        // input volumes. In convnets we do :(
        if outSx == 1 && outSy == 1 {
            for i in 0 ..< N {
                
                let ix = i * groupSize // base index offset
                var a = V.w[ix]
                var ai = 0
                for j in 1 ..< groupSize {
                    
                    let a2 = V.w[ix+j]
                    if a2 > a {
                        a = a2
                        ai = j
                    }
                }
                V2.w[i] = a
                switches[i] = ix + ai
            }
        } else {
            var n=0 // counter for switches
            for x in 0 ..< V.sx {
                
                for y in 0 ..< V.sy {
                    
                    for i in 0 ..< N {
                        
                        let ix = i * groupSize
                        var a = V.get(x: x, y: y, d: ix)
                        var ai = 0
                        for j in 1 ..< groupSize {
                            
                            let a2 = V.get(x: x, y: y, d: ix+j)
                            if a2 > a {
                                a = a2
                                ai = j
                            }
                        }
                        V2.set(x: x, y: y, d: i, v: a)
                        switches[n] = ix + ai
                        n += 1
                    }
                }
            }
            
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
        let N = outDepth
        V.dw = ArrayUtils.zerosFloat(V.w.count) // zero out gradient wrt data
        
        // pass the gradient through the appropriate switch
        if outSx == 1 && outSy == 1 {
            for i in 0 ..< N {
                
                let chainGrad = V2.dw[i]
                V.dw[switches[i]] = chainGrad
            }
        } else {
            // bleh okay, lets do this the hard way
            var n=0 // counter for switches
            for x in 0 ..< V2.sx {
                
                for y in 0 ..< V2.sy {
                    
                    for i in 0 ..< N {
                        
                        let chainGrad = V2.getGrad(x: x, y: y, d: i)
                        V.setGrad(x: x, y: y, d: switches[n], v: chainGrad)
                        n += 1
                    }
                }
            }
        }
    }
    
    public func getParamsAndGrads() -> [ParamsAndGrads] {
        return []
    }
    
    public func assignParamsAndGrads(_ paramsAndGrads: [ParamsAndGrads]) ->() {
        
    }
}
