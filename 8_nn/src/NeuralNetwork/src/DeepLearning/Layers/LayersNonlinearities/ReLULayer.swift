
import Foundation

// Implements ReLU nonlinearity elementwise
// x -> max(0, x)
// the output is in [0, inf)

public struct ReluLayerOpt: LayerInOptProtocol {
    public var inSx = 0
    public var inSy = 0
    public var inDepth = 0
}

public class ReluLayer: HiddenLayer {
    public var outWidth: Int
    public var outHeight: Int
    public var outDepth: Int
    public var inActivations: Volume?
    public var outActivations: Volume?
    
    public init(opt: ReluLayerOpt) {
        // computed
        outWidth = opt.inSx
        outHeight = opt.inSy
        outDepth = opt.inDepth
    }
    
    public func forward(_ volume: inout Volume, isTraining: Bool) -> Volume {
        inActivations = volume
        let outActivations = volume.clone()
        
        for i in 0 ..< outActivations.values.count {
            if outActivations.values[i] < 0 {
                // threshold at 0
                outActivations.values[i] = 0
            }
        }
        self.outActivations = outActivations
        return outActivations
    }
    
    public func backward() {
        guard let inActivations = inActivations,
            let outActivations = outActivations else {
                fatalError("inActivations or outActivations is nil")
        }
        
        let n = inActivations.values.count
        // zero out gradient wrt data
        inActivations.gradients = ArrayUtils.zerosFloat(n)
        for i in 0 ..< n {
            if outActivations.values[i] <= 0 {
                // threshold
                inActivations.gradients[i] = 0
            } else {
                inActivations.gradients[i] = outActivations.gradients[i]
            }
        }
    }
}
