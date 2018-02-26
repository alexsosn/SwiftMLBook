
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
    public var outWidth = 0
    public var outHeight = 0
    public var outDepth = 0
    
    public var dropped: [Bool]
    public var dropProb: Float
    
    public var inActivations: Volume?
    public var outActivations: Volume?
    
    public init(opt: DropoutLayerOpt) {
        // computed
        outWidth = opt.inSx
        outHeight = opt.inSy
        outDepth = opt.inDepth
        dropProb = opt.dropProb
        dropped = [Bool](repeating: false, count: outWidth*outHeight*outDepth)
    }
    
    // default is prediction mode
    public func forward(_ volume: inout Volume, isTraining: Bool = false) -> Volume {
        inActivations = volume
        let outActivations = volume.clone()
        let n = volume.values.count
        
        dropped = ArrayUtils.zerosBool(n)
        
        if isTraining {
            // do dropout
            for i in 0 ..< n {
                if Random.Uniform.float()<dropProb {
                    outActivations.values[i]=0
                    dropped[i] = true
                } // drop!
            }
        } else {
            // scale the activations during prediction
            for i in 0 ..< n {
                outActivations.values[i]*=(1-dropProb)
            }
        }
        self.outActivations = outActivations
        return outActivations
    }
    
    public func backward() {
        guard let volume = inActivations,
            let chainGrad = outActivations else { return }
        
        let n = volume.values.count
        volume.gradients = ArrayUtils.zerosFloat(n)
        // zero out gradient wrt data
        for i in 0 ..< n {
            
            if !(dropped[i]) {
                volume.gradients[i] = chainGrad.gradients[i] // copy over the gradient
            }
        }
    }
}
