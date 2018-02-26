
import Foundation

// Implements Sigmoid nnonlinearity elementwise
// x -> 1/(1+e^(-x))
// so the output is between 0 and 1.

public struct SigmoidLayerOpt: LayerInOptProtocol {
    public var inSx = 0
    public var inSy = 0
    public var inDepth = 0
}

public class SigmoidLayer: HiddenLayer {
    public var outWidth: Int
    public var outHeight: Int
    public var outDepth: Int
    public var inActivations: Volume?
    public var outActivations: Volume?
    
    public init(opt: SigmoidLayerOpt){
        
        // computed
        outWidth = opt.inSx
        outHeight = opt.inSy
        outDepth = opt.inDepth
    }
    // http://memkite.com/blog/2014/12/15/data-parallel-programming-with-metal-and-swift-for-iphoneipad-gpu/
    public func forward(_ volume: inout Volume, isTraining: Bool) -> Volume {
        self.inActivations = volume
        let outActivations = volume.cloneAndZero()
        for i in 0 ..< volume.values.count {
            outActivations.values[i] = Float(1.0)/(1.0+exp(-volume.values[i]))
        }
        self.outActivations = outActivations
        return outActivations
    }
    
    public func backward() {
        guard let inActivations = inActivations,
            let outActivations = outActivations
            else {
                fatalError("inActivations or outActivations is nil")
        }
        let n = inActivations.values.count
        inActivations.gradients = ArrayUtils.zerosFloat(n)
        // zero out gradient wrt data
        for i in 0 ..< n {
            inActivations.gradients[i] =  outActivations.values[i] * (1.0 - outActivations.values[i]) * outActivations.gradients[i]
        }
    }
}
