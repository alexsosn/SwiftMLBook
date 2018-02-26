// a bit experimental layer for now. I think it works but I'm not 100%
// the gradient check is a bit funky. I'll look into this a bit later.
// Local Response Normalization in window, along depths of volumes
import Foundation

public struct LocalResponseNormalizationLayerOpt: LayerInOptProtocol {
    public var k: Float
    public var n: Float
    public var α: Float
    public var β: Float
    public var inSx: Int
    public var inSy: Int
    public var inDepth: Int
}

public class LocalResponseNormalizationLayer: HiddenLayer {
    
    public var k: Float = 0.0
    public var n: Float = 0
    public var α: Float = 0.0
    public var β: Float = 0.0
    public var outWidth = 0
    public var outHeight = 0
    public var outDepth = 0
    public var inActivations: Volume?
    public var outActivations: Volume?
    public var S_cache_: Volume?
    
    public init(opt: LocalResponseNormalizationLayerOpt) {
        
        // required
        k = opt.k
        n = opt.n
        α = opt.α
        β = opt.β
        
        // computed
        outWidth = opt.inSx
        outHeight = opt.inSy
        outDepth = opt.inDepth
        
        // checks
        if n.truncatingRemainder(dividingBy: 2) == 0 { print("WARNING n should be odd for LRN layer"); }
    }
    
    public func forward(_ volume: inout Volume, isTraining: Bool) -> Volume {
        inActivations = volume
        
        let outActivations = volume.cloneAndZero()
        S_cache_ = volume.cloneAndZero()
        let n2 = Int(floor(n/2))
        for x in 0 ..< volume.width {
            for y in 0 ..< volume.height {
                for i in 0 ..< volume.depth {
                    let ai = volume.get(x: x, y: y, d: i)
                    
                    // normalize in a window of size n
                    var den = Float(0.0)
                    for j in max(0, i-n2) ... min(i+n2, volume.depth-1) {
                        let aa = volume.get(x: x, y: y, d: j)
                        den += aa*aa
                    }
                    den *= α / n
                    den += k
                    S_cache_!.set(x: x, y: y, d: i, v: den) // will be useful for backprop
                    den = pow(den, β)
                    outActivations.set(x: x, y: y, d: i, v: ai/den)
                }
            }
        }
        
        self.outActivations = outActivations
        return outActivations // dummy identity function for now
    }
    
    public func backward() {
        // evaluate gradient wrt data
        guard let volume = inActivations,
            let outActivations = outActivations,
            let S_cache_ = S_cache_ else {
                fatalError("inActivations or outActivations or S_cache_ is nil")
        }
        
        volume.gradients = ArrayUtils.zerosFloat(volume.values.count)
        // zero out gradient wrt data
        //        let A = outActivations // computed in forward pass
        
        let n2 = Int(floor(n/2))
        for x in 0 ..< volume.width {
            for y in 0 ..< volume.height {
                for i in 0 ..< volume.depth {
                    let chainGrad = outActivations.getGrad(x: x, y: y, d: i)
                    let S = S_cache_.get(x: x, y: y, d: i)
                    let SB = pow(S, β)
                    let SB2 = SB*SB
                    
                    // normalize in a window of size n
                    for j in max(0, i-n2) ... min(i+n2, volume.depth-1) {
                        let aj = volume.get(x: x, y: y, d: j)
                        var g = -aj*β*pow(S, β-1)*α/n*2*aj
                        if j==i {
                            g += SB
                        }
                        g /= SB2
                        g *= chainGrad
                        volume.addGrad(x: x, y: y, d: j, v: g)
                    }
                }
            }
        }
    }
}

