
import Foundation

// Implements Maxout nnonlinearity that computes
// x -> max(x)
// where x is a vector of size groupSize. Ideally of course,
// the input size should be exactly divisible by groupSize

public struct MaxoutLayerOpt: LayerInOptProtocol {
    
    public var inSx = 1
    public var inSy = 1
    public var inDepth = 1
    
    public var groupSize: Int
    
    public init (groupSize: Int?) {
        self.groupSize = groupSize ?? 2
    }
}

public class MaxoutLayer: HiddenLayer {
    public var groupSize: Int
    public var outWidth: Int
    public var outHeight: Int
    public var outDepth: Int
    public var inActivations: Volume?
    public var outActivations: Volume?
    public var switches: [Int]
    
    public init(opt: MaxoutLayerOpt){
        
        // required
        groupSize = opt.groupSize
        
        // computed
        outWidth = opt.inSx
        outHeight = opt.inSy
        outDepth = opt.inDepth / groupSize // WARNING: floor was here
        
        switches = ArrayUtils.zerosInt(outWidth*outHeight*outDepth) // useful for backprop
    }
    
    public func forward(_ volume: inout Volume, isTraining: Bool) -> Volume {
        inActivations = volume
        let depth = outDepth
        let outActivations = Volume(width: outWidth, height: outHeight, depth: outDepth, fill: 0.0)
        
        // optimization branch. If we're operating on 1D arrays we dont have
        // to worry about keeping track of x,y,d coordinates inside
        // input volumes. In convnets we do :(
        if outWidth == 1 && outHeight == 1 {
            for i in 0 ..< depth {
                
                let ix = i * groupSize // base index offset
                var a = volume.values[ix]
                var ai = 0
                for j in 1 ..< groupSize {
                    
                    let a2 = volume.values[ix+j]
                    if a2 > a {
                        a = a2
                        ai = j
                    }
                }
                outActivations.values[i] = a
                switches[i] = ix + ai
            }
        } else {
            var n=0 // counter for switches
            for x in 0 ..< volume.width {
                for y in 0 ..< volume.height {
                    for i in 0 ..< depth {
                        let ix = i * groupSize
                        var a = volume.get(x: x, y: y, d: ix)
                        var ai = 0
                        for j in 1 ..< groupSize {
                            let a2 = volume.get(x: x, y: y, d: ix+j)
                            if a2 > a {
                                a = a2
                                ai = j
                            }
                        }
                        outActivations.set(x: x, y: y, d: i, v: a)
                        switches[n] = ix + ai
                        n += 1
                    }
                }
            }
        }
        self.outActivations = outActivations
        return outActivations
    }
    
    public func backward() {
        guard let volume = inActivations,
            let volume2 = outActivations else {
                fatalError("inActivations or outActivations is nil")
        }
        let depth = outDepth
        volume.gradients = ArrayUtils.zerosFloat(volume.values.count)
        // zero out gradient wrt data
        
        // pass the gradient through the appropriate switch
        if outWidth == 1 && outHeight == 1 {
            for i in 0 ..< depth {
                let chainGrad = volume2.gradients[i]
                volume.gradients[switches[i]] = chainGrad
            }
        } else {
            // bleh okay, lets do this the hard way
            var n = 0 // counter for switches
            for x in 0 ..< volume2.width {
                for y in 0 ..< volume2.height {
                    for i in 0 ..< depth {
                        let chainGrad = volume2.getGrad(x: x, y: y, d: i)
                        volume.setGrad(x: x, y: y, d: switches[n], v: chainGrad)
                        n += 1
                    }
                }
            }
        }
    }
}
