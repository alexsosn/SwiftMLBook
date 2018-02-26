import Foundation

public struct InputLayerOpt: LayerOutOptProtocol {
    public var outSx: Int
    public var outSy: Int
    public var outDepth: Int
    
    public init(outSx: Int,
                outSy: Int,
                outDepth: Int) {
        self.outSx = outSx
        self.outSy = outSy
        self.outDepth = outDepth
    }
}

public class InputLayer: HiddenLayer {
    
    public var outDepth: Int
    public var outSx: Int
    public var outSy: Int
    
    public var inAct: Volume?
    public var outAct: Volume?
    
    public init(opt: InputLayerOpt) {
        
        // required: depth
        outDepth = opt.outDepth
        
        // optional: default these dimensions to 1
        outSx = opt.outSx
        outSy = opt.outSy
    }
    
    public func forward(_ V: inout Volume, isTraining: Bool) -> Volume {
        inAct = V
        outAct = V
        return outAct! // simply identity function for now
    }
    
    public func backward() -> () {}
    
    public func getParamsAndGrads() -> [ParamsAndGrads] {
        return []
    }
    
    public func assignParamsAndGrads(_ paramsAndGrads: [ParamsAndGrads])->() {
        
    }
}
