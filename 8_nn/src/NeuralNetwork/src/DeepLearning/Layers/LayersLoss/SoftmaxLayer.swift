
import Foundation

// This is a classifier, with N discrete classes from 0 to N-1
// it gets a stream of N incoming numbers and computes the softmax
// function (exponentiate and normalize to sum to 1 as probabilities should)

public struct SoftmaxLayerOpt: LayerInOptProtocol, ClassificationLayerOptProtocol {
    public var numClasses: Int
    
    // This properties will be set by the Net class.
    public var inSx = 0
    public var inSy = 0
    public var inDepth = 0
    
    public init (numClasses: Int) {
        self.numClasses = numClasses
    }
}

public class SoftmaxLayer: LossLayer {
    
    public var numInputs: Int
    public var outDepth: Int
    public var outWidth: Int
    public var outHeight: Int
    public var inActivations: Volume?
    public var outActivations: Volume?
    
    public var es: [Float] = []
    
    public init(opt: SoftmaxLayerOpt) {
        // computed
        numInputs = opt.numInputs()
        outDepth = numInputs
        outWidth = 1
        outHeight = 1
    }
    
    public func forward(_ volume: inout Volume, isTraining: Bool) -> Volume {
        inActivations = volume
        
        let outActivations = Volume(width: 1, height: 1, depth: outDepth, fill: 0.0)
        
        // compute max activation
        var a_s = volume.values
        let amax = volume.values.max()!
        
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
            outActivations.values[i] = es[i]
        }
        
        self.es = es // save these for backprop
        self.outActivations = outActivations
        return outActivations
    }
    
    public func backward(_ y: Int) -> Float {
        
        // compute and accumulate gradient wrt weights and bias of this layer
        guard let x = inActivations else {
            fatalError("inActivations is nil")
        }
        x.gradients = ArrayUtils.zerosFloat(x.values.count) // zero out the gradient of input Volume
        
        for i in 0 ..< outDepth {
            
            let indicator: Float = i == y ? 1.0 : 0.0
            let mul = -(indicator - es[i])
            x.gradients[i] = mul
        }
        // loss is the class negative log likelihood
        return -log(es[y])
    }
}
