
import Foundation

// Implements Tanh nnonlinearity elementwise
// x -> tanh(x)
// so the output is between -1 and 1.

public struct TanhLayerOpt: LayerInOptProtocol {
    public var inSx = 0
    public var inSy = 0
    public var inDepth = 0
}

public class TanhLayer: HiddenLayer {
    public var outWidth: Int
    public var outHeight: Int
    public var outDepth: Int
    public var inActivations: Volume?
    public var outActivations: Volume?
    
    public init(opt: TanhLayerOpt) {
        
        // computed
        outWidth = opt.inSx
        outHeight = opt.inSy
        outDepth = opt.inDepth
    }
    
    public func forward(_ volume: inout Volume, isTraining: Bool) -> Volume {
        inActivations = volume
        let outActivations = volume.cloneAndZero()
        for i in 0 ..< volume.values.count {
            outActivations.values[i] = tanh(volume.values[i])
        }
        self.outActivations = outActivations
        return outActivations
    }
    
    public func backward() {
        guard let volume = inActivations,
            let volume2 = outActivations else {
                fatalError("inActivations or outActivations is nil")
        }
        let n = volume.values.count
        volume.gradients = ArrayUtils.zerosFloat(n)
        // zero out gradient wrt data
        for i in 0 ..< n {
            let v2wi = volume2.values[i]
            volume.gradients[i] = (1.0 - v2wi * v2wi) * volume2.gradients[i]
        }
    }
}
