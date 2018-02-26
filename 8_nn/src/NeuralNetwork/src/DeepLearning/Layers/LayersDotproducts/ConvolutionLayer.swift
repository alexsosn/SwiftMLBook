
import Foundation

public struct ConvolutionLayerOpt: LayerInOptProtocol, LayerOptActivationProtocol {
    public var width: Int
    public var height: Int
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
    
    
    public init(width: Int,
                height: Int? = nil,
                filters: Int,
                inDepth: Int = 0,
                inSx: Int = 0,
                inSy: Int = 0,
                stride: Int = 1,
                pad: Int = 0,
                l1DecayMul: Float = 0.0,
                l2DecayMul: Float = 1.0,
                biasPref: Float = 0.0,
                activation: ActivationType = .undefined) {
        self.width = width
        self.height = height ?? width
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
public class ConvolutionLayer: DotProductLayer {
    public var outDepth: Int
    public var width: Int
    public var height: Int
    public var inDepth: Int
    public var inSx: Int
    public var inSy: Int
    public var stride: Int
    public var pad: Int
    public var l1DecayMul: Float
    public var l2DecayMul: Float
    public var outWidth: Int
    public var outHeight: Int
    public var filters: [Volume]
    public var biases: Volume
    public var inActivations: Volume?
    public var outActivations: Volume?
    
    public init(opt: ConvolutionLayerOpt) {
        
        // required
        outDepth = opt.filters
        width = opt.width // filter size. Should be odd if possible, it's cleaner.
        inDepth = opt.inDepth
        inSx = opt.inSx
        inSy = opt.inSy
        
        // optional
        height = opt.height
        stride = opt.stride // stride at which we apply filters to input volume
        pad = opt.pad // amount of 0 padding to add around borders of input volume
        
        l1DecayMul = opt.l1DecayMul
        l2DecayMul = opt.l2DecayMul
        
        // computed
        // note we are doing floor, so if the strided convolution of the filter doesn't fit into the input
        // volume exactly, the output volume will be trimmed and not contain the (incomplete) computed
        // final application.
        outWidth = Int(floor(Float(inSx + pad * 2 - width) / Float(stride + 1)))
        outHeight = Int(floor(Float(inSy + pad * 2 - height) / Float(stride + 1)))
        
        // initializations
        let bias = opt.biasPref
        filters = (0..<outDepth).map{_ in Volume(width: opt.width, height: opt.height, depth: opt.inDepth)}
        biases = Volume(width: 1, height: 1, depth: outDepth, fill: bias)
    }
    
    public func forward(_ volume: inout Volume, isTraining: Bool) -> Volume {
        // optimized code by @mdda that achieves 2x speedup over previous version
        
        inActivations = volume
        let outActivations = Volume(width: outWidth, height: outHeight, depth: outDepth, fill: 0.0)
        
        let xyStride = stride
        
        for d in 0 ..< outDepth {
            let filter = filters[d]
            var x = -pad
            var y = -pad
            
            for ay in 0 ..< outHeight {
                y+=xyStride // xy_stride
                x = -pad
                
                for ax in 0 ..< outWidth {  // xy_stride
                    x+=xyStride
                    // convolve centered at this particular location
                    var a: Float = Float(0.0)
                    
                    for fy in 0 ..< filter.height {
                        let oy = y+fy // coordinates in the original input array coordinates
                        
                        for fx in 0 ..< filter.width {
                            let ox = x+fx
                            if oy>=0 && oy<volume.height && ox>=0 && ox<volume.width {
                                
                                for fd in 0 ..< filter.depth {
                                    // avoid function call overhead (x2) for efficiency, compromise modularity :(
                                    a += filter.values[((filter.width * fy)+fx)*filter.depth+fd]
                                        * volume.values[((volume.width * oy)+ox)*volume.depth+fd]
                                }
                            }
                        }
                    }
                    a += biases.values[d]
                    outActivations.set(x: ax, y: ay, d: d, v: a)
                }
            }
        }
        self.outActivations = outActivations
        return outActivations
    }
    
    public func backward() {
        
        guard let volume = inActivations,
            let outActivations = outActivations
            else {
                return
        }
        volume.gradients = ArrayUtils.zerosFloat(volume.values.count)
        // zero out gradient wrt bottom data, we're about to fill it
        
        let xyStride = stride
        
        for d in 0 ..< outDepth {
            
            let filter = filters[d]
            var x = -pad
            var y = -pad
            for ay in 0 ..< outHeight {
                y+=xyStride
                x = -pad
                for ax in 0 ..< outWidth {
                    x+=xyStride
                    
                    // convolve centered at this particular location
                    let chainGrad = outActivations.getGrad(x: ax, y: ay, d: d) // gradient from above, from chain rule
                    for fy in 0 ..< filter.height {
                        
                        let oy = y+fy // coordinates in the original input array coordinates
                        for fx in 0 ..< filter.width {
                            
                            let ox = x+fx
                            if oy>=0 && oy<volume.height && ox>=0 && ox<volume.width {
                                for fd in 0 ..< filter.depth {
                                    
                                    // avoid function call overhead (x2) for efficiency, compromise modularity :(
                                    let ix1 = ((volume.width * oy)+ox)*volume.depth+fd
                                    let ix2 = ((filter.width * fy)+fx)*filter.depth+fd
                                    filter.gradients[ix2] += volume.values[ix1]*chainGrad
                                    volume.gradients[ix1] += filter.values[ix2]*chainGrad
                                }
                            }
                        }
                    }
                    biases.gradients[d] += chainGrad
                }
            }
            filters[d] = filter
        }
    }
}
