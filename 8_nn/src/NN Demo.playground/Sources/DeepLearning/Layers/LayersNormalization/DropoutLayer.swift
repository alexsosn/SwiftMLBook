
// An inefficient dropout layer
// Note this is not most efficient implementation since the layer before
// computed all these activations and now we're just going to drop them :(
// same goes for backward pass. Also, if we wanted to be efficient at test time
// we could equivalently be clever and upscale during train and copy pointers during test
// todo: make more efficient.
import Foundation

public struct DropoutLayerOpt: LayerInOptProtocol {
    public var inDepth: Int
    public var inSx: Int
    public var inSy: Int
    public var dropProb: Float
    
    public init(inDepth: Int = 0,
                inSx: Int = 0,
                inSy: Int = 0,
                dropProb: Float? = 0.5) {
        self.inDepth = inDepth
        self.inSx = inSx
        self.inSy = inSy
        self.dropProb = dropProb ?? 0.0
    }
}

public class DropoutLayer: HiddenLayer {
    public var outSx: Int = 0
    public var outSy: Int = 0
    public var outDepth: Int = 0
    
    public var dropped: [Bool]
    public var dropProb: Float
    
    public var inAct: Volume?
    public var outAct: Volume?
    
    public init(opt: DropoutLayerOpt) {
        // computed
        outSx = opt.inSx
        outSy = opt.inSy
        outDepth = opt.inDepth
        dropProb = opt.dropProb
        dropped = [Bool](repeating: false, count: outSx*outSy*outDepth)
    }
    
    // default is prediction mode
    public func forward(_ V: inout Volume, isTraining: Bool = false) -> Volume {
        inAct = V
        let V2 = V.clone()
        let N = V.w.count
        
        dropped = ArrayUtils.zerosBool(N)
        
        if isTraining {
            // do dropout
            for i in 0 ..< N {
                
                if Random.Uniform.float()<dropProb {
                    V2.w[i]=0
                    dropped[i] = true
                } // drop!
            }
        } else {
            // scale the activations during prediction
            for i in 0 ..< N {
                V2.w[i]*=(1-dropProb)
            }
        }
        outAct = V2
        return outAct! // dummy identity function for now
    }
    
    public func backward() -> () {
        
        guard let V = inAct, // we need to set dw of this
            let chainGrad = outAct
            else { return }
        let N = V.w.count
        V.dw = ArrayUtils.zerosFloat(N) // zero out gradient wrt data
        for i in 0 ..< N {
            
            if !(dropped[i]) {
                V.dw[i] = chainGrad.dw[i] // copy over the gradient
            }
        }
    }
    
    public func getParamsAndGrads() -> [ParamsAndGrads] {
        return []
    }
    
    public func assignParamsAndGrads(_ paramsAndGrads: [ParamsAndGrads]) ->() {
        
    }
}
