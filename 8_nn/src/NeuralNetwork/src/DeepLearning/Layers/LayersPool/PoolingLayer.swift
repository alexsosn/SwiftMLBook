import Foundation

public struct PoolingLayerOpt: LayerInOptProtocol {
    public var width: Int
    public var height: Int
    public var inDepth: Int
    public var inSx: Int
    public var inSy: Int
    public var stride: Int
    
    // amount of 0 padding to add around borders of input volume
    public var pad: Int
    
    public init(width: Int,
                height: Int? = nil,
                inDepth: Int = 0,
                inSx: Int = 0,
                inSy: Int = 0,
                stride: Int = 2,
                pad: Int = 0) {
        self.width = width
        self.height = height ?? width
        self.inDepth = inDepth
        self.inSx = inSx
        self.inSy = inSy
        self.stride = stride
        self.pad = pad
    }
}

public class PoolingLayer: HiddenLayer {
    public var width: Int
    public var height: Int
    public var inDepth: Int
    public var inSx: Int
    public var inSy: Int
    public var stride: Int
    public var pad: Int
    public var outDepth: Int
    public var outWidth: Int
    public var outHeight: Int
    public var switchx: [Int]
    public var switchy: [Int]
    public var inActivations: Volume?
    public var outActivations: Volume?
    
    public init(opt: PoolingLayerOpt){
        
        // required
        width = opt.width // filter size
        inDepth = opt.inDepth
        inSx = opt.inSx
        inSy = opt.inSy
        
        // optional
        height = opt.height
        stride = opt.stride
        pad = opt.pad
        
        // computed
        outDepth = inDepth
        outWidth = (inSx + pad * 2 - width) / stride + 1
        outHeight = (inSy + pad * 2 - height) / stride + 1
        
        // store switches for x,y coordinates for where the max comes from, for each output neuron
        switchx = ArrayUtils.zerosInt(outWidth*outHeight*outDepth)
        switchy = ArrayUtils.zerosInt(outWidth*outHeight*outDepth)
    }
    
    public func forward(_ volume: inout Volume, isTraining: Bool) -> Volume {
        precondition(width>0)
        precondition(height>0)
        
        inActivations = volume
        
        let outActivations = Volume(width: outWidth, height: outHeight, depth: outDepth, fill: 0.0)
        var n = 0 // a counter for switches
        
        for d in 0 ..< outDepth {
            var x = -pad
            
            for ax in 0 ..< outWidth {
                defer { x+=stride }
                var y = -pad
                
                for ay in 0 ..< outHeight {
                    defer { y+=stride }
                    
                    // convolve centered at this particular location
                    var a = Float(-99999.0) // hopefully small enough ;\
                    var winx = -1
                    var winy = -1
                    
                    for fx in 0 ..< width {
                        
                        for fy in 0 ..< height {
                            
                            let oy = y+fy
                            let ox = x+fx
                            if oy>=0 && oy<volume.height && ox>=0 && ox<volume.width {
                                let v = volume.get(x: ox, y: oy, d: d)
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
                    outActivations.set(x: ax, y: ay, d: d, v: a)
                }
            }
        }
        self.outActivations = outActivations
        return outActivations
    }
    
    public func backward() {
        // pooling layers have no parameters, so simply compute gradient wrt data here
        guard let inActivations = inActivations else {
            fatalError("inActivations is nil")
        }
        
        // computed in forward pass
        guard let outActivations = outActivations else {
            fatalError("outActivations is nil")
        }
        
        inActivations.gradients = ArrayUtils.zerosFloat(inActivations.values.count) // zero out gradient wrt data
        
        var n = 0
        for d in 0 ..< outDepth {
            
            var x = -pad
            
            for ax in 0 ..< outWidth {
                var y = -pad
                
                for ay in 0 ..< outHeight {
                    
                    let chainGrad = outActivations.getGrad(x: ax, y: ay, d: d)
                    inActivations.addGrad(x: switchx[n], y: switchy[n], d: d, v: chainGrad)
                    n += 1
                    y += stride
                }
                
                x += stride
            }
        }
    }
}


