import Foundation

public struct PoolingLayerOpt: LayerInOptProtocol {
    public var sx: Int
    public var sy: Int
    public var inDepth: Int
    public var inSx: Int
    public var inSy: Int
    public var stride: Int
    
    // amount of 0 padding to add around borders of input volume
    public var pad: Int
    
    public init(sx: Int,
                sy: Int? = nil,
                inDepth: Int = 0,
                inSx: Int = 0,
                inSy: Int = 0,
                stride: Int = 2,
                pad: Int = 0) {
        self.sx = sx
        self.sy = sy ?? sx
        self.inDepth = inDepth
        self.inSx = inSx
        self.inSy = inSy
        self.stride = stride
        self.pad = pad
    }
}

public class PoolingLayer: HiddenLayer {
    public var sx: Int
    public var sy: Int
    public var inDepth: Int
    public var inSx: Int
    public var inSy: Int
    public var stride: Int
    public var pad: Int
    public var outDepth: Int
    public var outSx: Int
    public var outSy: Int
    public var switchx: [Int]
    public var switchy: [Int]
    public var inAct: Volume?
    public var outAct: Volume?
    
    public init(opt: PoolingLayerOpt){
        
        // required
        sx = opt.sx // filter size
        inDepth = opt.inDepth
        inSx = opt.inSx
        inSy = opt.inSy
        
        // optional
        sy = opt.sy
        stride = opt.stride
        pad = opt.pad
        
        // computed
        outDepth = inDepth
        outSx = (inSx + pad * 2 - sx) / stride + 1
        outSy = (inSy + pad * 2 - sy) / stride + 1
        
        // store switches for x,y coordinates for where the max comes from, for each output neuron
        switchx = ArrayUtils.zerosInt(outSx*outSy*outDepth)
        switchy = ArrayUtils.zerosInt(outSx*outSy*outDepth)
    }
    
    public func forward(_ V: inout Volume, isTraining: Bool) -> Volume {
        precondition(sx>0)
        precondition(sy>0)
        
        inAct = V
        
        let A = Volume(sx: outSx, sy: outSy, depth: outDepth, c: 0.0)
        var n = 0 // a counter for switches
        
        for d in 0 ..< outDepth {
            var x = -pad
            
            for ax in 0 ..< outSx {
                defer { x+=stride }
                var y = -pad
                
                for ay in 0 ..< outSy {
                    defer { y+=stride }
                    
                    // convolve centered at this particular location
                    var a = Float(-99999.0) // hopefully small enough ;\
                    var winx = -1
                    var winy = -1
                    
                    for fx in 0 ..< sx {
                        
                        for fy in 0 ..< sy {
                            
                            let oy = y+fy
                            let ox = x+fx
                            if oy>=0 && oy<V.sy && ox>=0 && ox<V.sx {
                                let v = V.get(x: ox, y: oy, d: d)
                                // perform max pooling and store pointers to where
                                // the max came from. This will speed up backprop
                                // and can help make nice visualizations in future
                                if v > a {
                                    a = v;
                                    winx = ox;
                                    winy = oy;
                                }
                            }
                        }
                    }
                    switchx[n] = winx
                    switchy[n] = winy
                    n += 1
                    A.set(x: ax, y: ay, d: d, v: a)
                }
            }
        }
        outAct = A
        return outAct!
    }
    
    public func backward() -> () {
        // pooling layers have no parameters, so simply compute gradient wrt data here
        guard let inAct = inAct else {
            fatalError("inAct is nil")
        }
        
        // computed in forward pass
        guard let outAct = outAct else {
            fatalError("outAct is nil")
        }
        
        inAct.dw = ArrayUtils.zerosFloat(inAct.w.count) // zero out gradient wrt data
        
        var n = 0
        for d in 0 ..< outDepth {
            
            var x = -pad
            
            for ax in 0 ..< outSx {
                var y = -pad
                
                for ay in 0 ..< outSy {
                    
                    let chainGrad = outAct.getGrad(x: ax, y: ay, d: d)
                    inAct.addGrad(x: switchx[n], y: switchy[n], d: d, v: chainGrad)
                    n += 1
                    y += stride
                }
                
                x += stride
            }
        }
    }
    
    public func getParamsAndGrads() -> [ParamsAndGrads] {
        return []
    }
    
    public func assignParamsAndGrads(_ paramsAndGrads: [ParamsAndGrads]) ->() {
        
    }
}


