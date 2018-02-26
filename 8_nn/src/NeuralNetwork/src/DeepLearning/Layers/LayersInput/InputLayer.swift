import Foundation

public struct InputLayerOpt: LayerOutOptProtocol {
    public var outWidth: Int
    public var outHeight: Int
    public var outDepth: Int
}

public class InputLayer: Layer {
    
    public var outDepth: Int
    public var outWidth: Int
    public var outHeight: Int
    
    public var inActivations: Volume?
    public var outActivations: Volume?
    
    public init(opt: InputLayerOpt) {
        
        // required: depth
        outDepth = opt.outDepth
        
        // optional: default these dimensions to 1
        outWidth = opt.outWidth
        outHeight = opt.outHeight
    }
    
    public func forward(_ volume: inout Volume, isTraining: Bool) -> Volume {
        inActivations = volume
        outActivations = volume
        return volume // simply identity function for now
    }
}
