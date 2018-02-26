//
//  ConvolutionLayer.swift
//  ConvNetSwift
//
//  Created by Alex Sosnovshchenko on 2/17/17.
//  Copyright © 2017 Alex Sosnovshchenko. All rights reserved.
//


import Foundation

public struct ConvolutionLayerOpt: LayerInOptProtocol, LayerOptActivationProtocol {
    public var sx: Int
    public var sy: Int
    public var filters: Int
    public var inDepth: Int
    public var inSx: Int
    public var inSy: Int
    public var stride: Int
    public var pad: Int
    public var l1DecayMul: Float
    public var l2DecayMul: Float
    public var biasPref: Float
    public var activation: ActivationType
    
    
    public init(sx: Int,
                sy: Int? = nil,
                filters: Int,
                inDepth: Int = 0,
                inSx: Int = 0,
                inSy: Int = 0,
                stride: Int = 1,
                pad: Int = 0,
                l1DecayMul: Float = Float(0.0),
                l2DecayMul: Float = Float(1.0),
                biasPref: Float = Float(0.0),
                activation: ActivationType = .undefined) {
        self.sx = sx
        self.sy = sy ?? sx
        self.filters = filters
        self.inDepth = inDepth
        self.inSx = inSx
        self.inSy = inSy
        self.stride = stride
        self.pad = pad
        self.l1DecayMul = l1DecayMul
        self.l2DecayMul = l2DecayMul
        self.biasPref = biasPref
        self.activation = activation
    }
}

// - ConvolutionLayer does convolutions (so weight sharing spatially)
public class ConvolutionLayer: HiddenLayer {
    public var outDepth: Int
    public var sx: Int
    public var sy: Int
    public var inDepth: Int
    public var inSx: Int
    public var inSy: Int
    public var stride: Int
    public var pad: Int
    public var l1DecayMul: Float
    public var l2DecayMul: Float
    public var outSx: Int
    public var outSy: Int
    public var filters: [Volume]
    public var biases: Volume
    public var inAct: Volume?
    public var outAct: Volume?
    
    public init(opt: ConvolutionLayerOpt) {
        
        // required
        outDepth = opt.filters
        sx = opt.sx // filter size. Should be odd if possible, it's cleaner.
        inDepth = opt.inDepth
        inSx = opt.inSx
        inSy = opt.inSy
        
        // optional
        sy = opt.sy
        stride = opt.stride // stride at which we apply filters to input volume
        pad = opt.pad // amount of 0 padding to add around borders of input volume
        
        l1DecayMul = opt.l1DecayMul
        l2DecayMul = opt.l2DecayMul
        
        // computed
        // note we are doing floor, so if the strided convolution of the filter doesn't fit into the input
        // volume exactly, the output volume will be trimmed and not contain the (incomplete) computed
        // final application.
        outSx = Int(floor(Float(inSx + pad * 2 - sx) / Float(stride + 1)))
        outSy = Int(floor(Float(inSy + pad * 2 - sy) / Float(stride + 1)))
        
        // initializations
        let bias = opt.biasPref
        filters = []
        for _ in 0..<outDepth {
            filters.append(Volume(sx: sx, sy: sy, depth: inDepth))
        }
        biases = Volume(sx: 1, sy: 1, depth: outDepth, c: bias)
    }
    
    public func forward(_ V: inout Volume, isTraining: Bool) -> Volume {
        // optimized code by @mdda that achieves 2x speedup over previous version
        
        inAct = V
        let A = Volume(sx: outSx, sy: outSy, depth: outDepth, c: 0.0)
        
        let V_sx = V.sx
        let V_sy = V.sy
        let xy_stride = stride
        
        for d in 0 ..< outDepth {
            let f = filters[d]
            var x = -pad
            var y = -pad
            
            for ay in 0 ..< outSy {
                y+=xy_stride // xy_stride
                x = -pad
                
                for ax in 0 ..< outSx {  // xy_stride
                    x+=xy_stride
                    // convolve centered at this particular location
                    var a: Float = Float(0.0)
                    
                    for fy in 0 ..< f.sy {
                        let oy = y+fy // coordinates in the original input array coordinates
                        
                        for fx in 0 ..< f.sx {
                            let ox = x+fx
                            if oy>=0 && oy<V_sy && ox>=0 && ox<V_sx {
                                
                                for fd in 0 ..< f.depth {
                                    // avoid function call overhead (x2) for efficiency, compromise modularity :(
                                    a += f.w[((f.sx * fy)+fx)*f.depth+fd] * V.w[((V_sx * oy)+ox)*V.depth+fd]
                                }
                            }
                        }
                    }
                    a += biases.w[d]
                    A.set(x: ax, y: ay, d: d, v: a)
                }
            }
        }
        outAct = A
        return outAct!
    }
    
    public func backward() -> () {
        
        guard
            let V = inAct,
            let outAct = outAct
            else {
                return
        }
        V.dw = ArrayUtils.zerosFloat(V.w.count) // zero out gradient wrt bottom data, we're about to fill it
        
        let V_sx = V.sx
        let V_sy = V.sy
        let xy_stride = stride
        
        for d in 0 ..< outDepth {
            
            let f = filters[d]
            var x = -pad
            var y = -pad
            for ay in 0 ..< outSy {
                y+=xy_stride
                x = -pad
                for ax in 0 ..< outSx {
                    x+=xy_stride
                    
                    // convolve centered at this particular location
                    let chainGrad = outAct.getGrad(x: ax, y: ay, d: d) // gradient from above, from chain rule
                    for fy in 0 ..< f.sy {
                        
                        let oy = y+fy // coordinates in the original input array coordinates
                        for fx in 0 ..< f.sx {
                            
                            let ox = x+fx
                            if oy>=0 && oy<V_sy && ox>=0 && ox<V_sx {
                                for fd in 0 ..< f.depth {
                                    
                                    // avoid function call overhead (x2) for efficiency, compromise modularity :(
                                    let ix1 = ((V_sx * oy)+ox)*V.depth+fd
                                    let ix2 = ((f.sx * fy)+fx)*f.depth+fd
                                    f.dw[ix2] += V.w[ix1]*chainGrad
                                    V.dw[ix1] += f.w[ix2]*chainGrad
                                }
                            }
                        }
                    }
                    biases.dw[d] += chainGrad
                }
            }
            filters[d] = f
        }
    }
    
    public func getParamsAndGrads() -> [ParamsAndGrads] {
        var response: [ParamsAndGrads] = []
        for i in 0 ..< outDepth {
            
            response.append(ParamsAndGrads(
                params: &filters[i].w,
                grads: &filters[i].dw,
                l1DecayMul: l1DecayMul,
                l2DecayMul: l2DecayMul))
        }
        response.append(ParamsAndGrads(
            params: &biases.w,
            grads: &biases.dw,
            l1DecayMul: 0.0,
            l2DecayMul: 0.0))
        return response
    }
    
    public func assignParamsAndGrads(_ paramsAndGrads: [ParamsAndGrads]) ->() {
        assert(filters.count + 1 == paramsAndGrads.count)
        
        for i in 0 ..< outDepth {
            filters[i].w = paramsAndGrads[i].params
            filters[i].dw = paramsAndGrads[i].grads
        }
        biases.w = paramsAndGrads.last!.params
        biases.dw = paramsAndGrads.last!.grads
    }
}
