// Volume is the basic building block of all data in a net.
// it is essentially just a 3D volume of numbers, with a
// width (sx), height (sy), and depth (depth).
// It is used to hold data for all filters, all volumes,
// all weights, and also stores all gradients w.r.t.
// the data. c is optionally a value to initialize the volume
// with. If c is missing, fills the Volume with random numbers.

import Foundation

public class Volume {
    public var sx: Int = 1
    public var sy: Int = 1
    public var depth: Int = 0
    public var w: [Float] = []
    public var dw: [Float] = []
    
    public convenience init (array: [Float]) {
        self.init()

        // we were given a list in sx, assume 1D volume and fill it up
        sx = 1
        sy = 1
        depth = array.count

        w = ArrayUtils.zerosFloat(depth)
        dw = ArrayUtils.zerosFloat(depth)
        for i in 0 ..< depth {
            w[i] = array[i]
        }
    }
    
    public convenience init(width sx: Int, height sy: Int, depth: Int, array: [Float]) {
        self.init()
        assert(array.count==sx*sy*depth)
        self.sx = sx
        self.sy = sy
        self.depth = depth
        w = array
    }
    
    public convenience init(sx: Int, sy: Int, depth: Int) {
        self.init(width: sx, height: sy, depth: depth, c: nil)
    }
    
    public convenience init(sx: Int, sy: Int, depth: Int, c: Float) {
        self.init(width: sx, height: sy, depth: depth, c: c)
    }
    
    public convenience init(width sx: Int, height sy: Int, depth: Int, c: Float?) {
        self.init()
        // we were given dimensions of the vol
        self.sx = sx
        self.sy = sy
        self.depth = depth
        let n = sx*sy*depth
        w = ArrayUtils.zerosFloat(n)
        dw = ArrayUtils.zerosFloat(n)
        if let c = c {
            for i in 0 ..< n {
                w[i] = c
            }
        } else {
            // weight normalization is done to equalize the output
            // variance of every neuron, otherwise neurons with a lot
            // of incoming connections have outputs of larger variance
            let scale = sqrt(1.0/Float(sx*sy*depth))
            for i in 0 ..< n {

                w[i] = Random.Normal.float(0.0, std: scale)
            }
        }
    }
    
    public func get(x: Int, y: Int, d: Int) -> Float {
        let ix=((sx * y)+x)*depth+d
        return w[ix]
    }
    
    public func set(x: Int, y: Int, d: Int, v: Float) -> () {
        let ix=((sx * y)+x)*depth+d
        w[ix] = v
    }
    
    public func add(x: Int, y: Int, d: Int, v: Float) -> () {
        let ix=((sx * y)+x)*depth+d
        w[ix] += v
    }
    
    public func getGrad(x: Int, y: Int, d: Int) -> Float {
        let ix = ((sx * y)+x)*depth+d
        return dw[ix]
    }
    
    public func setGrad(x: Int, y: Int, d: Int, v: Float) -> () {
        let ix = ((sx * y)+x)*depth+d
        dw[ix] = v
    }
    
    public func addGrad(x: Int, y: Int, d: Int, v: Float) -> () {
        let ix = ((sx * y)+x)*depth+d
        dw[ix] += v
    }
    
    public func cloneAndZero() -> Volume {
        return Volume(sx: sx, sy: sy, depth: depth, c: 0.0)
    }
    
    public func clone() -> Volume {
        let V = Volume(sx: sx, sy: sy, depth: depth, c: 0.0)
        let n = w.count
        for i in 0 ..< n {
            V.w[i] = w[i]
        }
        return V
    }
    
    public func addFrom(_ V: Volume) {
        for k in 0 ..< w.count {
            w[k] += V.w[k]
        }
    }
    
    public func addFromScaled(_ V: Volume, a: Float) {
        for k in 0 ..< w.count {
            w[k] += a*V.w[k]
        }
    }
    
    public func setConst(_ a: Float) {
        for k in 0 ..< w.count {
            w[k] = a
        }
    }
    
    public func description() -> String {
        return "size: \(sx)*\(sy)*\(depth)\nw:\n\(w)\ndw:\n\(dw)"
    }
    
    public func debugDescription() -> String {
        return "size: \(sx)*\(sy)*\(depth)\nw:\n\(w)\ndw:\n\(dw)"
    }
}


