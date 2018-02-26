
import Foundation

public struct SVMLayerOpt: LayerInOptProtocol, ClassificationLayerOptProtocol {
    public var numClasses: Int
    
    // This properties will be set by the Net class.
    public var inSx = 0
    public var inSy = 0
    public var inDepth = 0
    
    public init (numClasses: Int) {
        self.numClasses = numClasses
    }
}

public class SVMLayer: LossLayer {
    public var numInputs: Int
    public var outDepth: Int
    public var outWidth: Int
    public var outHeight: Int
    public var inActivations: Volume?
    public var outActivations: Volume?
    
    public init(opt: SVMLayerOpt){
        // computed
        numInputs = opt.numInputs()
        outDepth = numInputs
        outWidth = 1
        outHeight = 1
    }
    
    public func forward(_ volume: inout Volume, isTraining: Bool) -> Volume {
        inActivations = volume
        outActivations = volume // nothing to do, output raw scores
        return volume
    }
    
    public func backward(_ y: Int) -> Float {
        
        // compute and accumulate gradient wrt weights and bias of this layer
        guard let x = inActivations else {
            fatalError("inActivations is nil")
        }
        
        x.gradients = ArrayUtils.zerosFloat(x.values.count) // zero out the gradient of input Volume
        
        // we're using structured loss here, which means that the score
        // of the ground truth should be higher than the score of any other
        // class, by a margin
        let yscore = x.values[y] // score of ground truth
        let margin = Float(1.0)
        var loss = Float(0.0)
        for i in 0 ..< outDepth {
            
            if y == i { continue }
            let ydiff = -yscore + x.values[i] + margin
            if ydiff > 0 {
                // violating dimension, apply loss
                x.gradients[i] += 1
                x.gradients[y] -= 1
                loss += ydiff
            }
        }
        inActivations = x
        return loss
    }
}

