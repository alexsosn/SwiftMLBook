
import Foundation

// implements an L2 regression cost layer,
// so penalizes \sum_i(||x_i - y_i||^2), where x is its input
// and y is the user-provided array of "correct" values.

public struct RegressionLayerOpt: LayerInOptProtocol {
    public var numNeurons: Int
    
    // This properties will be set by the Net class.
    public var inSx = 0
    public var inSy = 0
    public var inDepth = 0
    
    public init(numNeurons: Int = 1) {
        self.numNeurons = numNeurons
    }
}

public class RegressionLayer: LossLayer {
    public var numInputs: Int
    public var outDepth: Int
    public var outWidth: Int
    public var outHeight: Int
    public var inActivations: Volume?
    public var outActivations: Volume?
    
    public init(opt: RegressionLayerOpt) {
        
        // computed
        numInputs = opt.numInputs()
        outDepth = numInputs
        outWidth = 1
        outHeight = 1
    }
    
    public func forward(_ volume: inout Volume, isTraining: Bool) -> Volume {
        inActivations = volume
        outActivations = volume
        return volume // identity function
    }
    
    // y is a list here of size numInputs
    // or it can be a number if only one value is regressed
    // or it can be a struct {dim: i, val: x} where we only want to
    // regress on dimension i and asking it to have value x
    
    public func backward(_ y: [Float]) -> Float {
        
        // compute and accumulate gradient wrt weights and bias of this layer
        guard let x = inActivations else {
            fatalError("inActivations is nil")
        }
        x.gradients = ArrayUtils.zerosFloat(x.values.count) // zero out the gradient of input Volume
        var loss = Float(0.0)
        for i in 0 ..< outDepth {
            let dy = x.values[i] - y[i]
            x.gradients[i] = dy
            loss += 0.5*dy*dy
        }
        inActivations = x
        return loss
    }
    
    public func backward(_ y: Float) -> Float {
        // compute and accumulate gradient wrt weights and bias of this layer
        guard let x = inActivations else {
            fatalError("inActivations is nil")
        }
        x.gradients = ArrayUtils.zerosFloat(x.values.count) // zero out the gradient of input Volume
        var loss = Float(0.0)
        // lets hope that only one number is being regressed
        let dy = x.values[0] - y
        x.gradients[0] = dy
        loss += 0.5*dy*dy
        return loss
    }
    
    public func backward(_ y: Int) -> Float {
        return backward(Float(y))
    }
    
    public func backward(_ y: (dim: Int, val: Float)) -> Float {
        // compute and accumulate gradient wrt weights and bias of this layer
        guard let x = inActivations else {
            fatalError("inActivations is nil")
        }
        x.gradients = ArrayUtils.zerosFloat(x.values.count) // zero out the gradient of input Volume
        var loss = Float(0.0)
        // assume it is a struct with entries .dim and .val
        // and we pass gradient only along dimension dim to be equal to val
        let i = y.dim
        let yi = y.val
        let dy = x.values[i] - yi
        x.gradients[i] = dy
        loss += 0.5*dy*dy
        return loss
    }
}
