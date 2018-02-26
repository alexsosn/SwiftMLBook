// Volume is the basic building block of all data in a net.
// it is essentially just a 3D volume of numbers, with a
// width, height, and depth.
// It is used to hold data for all filters, all volumes,
// all weights, and also stores all gradients w.r.t.
// the data.

import Foundation

public class Volume {
    public var width = 1
    public var height = 1
    public var depth = 0
    
    public var values = [Float]()
    public var gradients = [Float]()
    
    public convenience init (array: [Float]) {
        self.init()

        // We were given a list in width, assume 1D volume and fill it up
        width = 1
        height = 1
        depth = array.count

        values = array
        gradients = ArrayUtils.zerosFloat(depth)
    }
    
    public convenience init(width: Int, height: Int, depth: Int, array: [Float]) {
        assert(array.count==width*height*depth)
        self.init(width: width, height: height, depth: depth)
        values = array
        gradients = ArrayUtils.zerosFloat(depth)
    }
    
    // `fill` is optionally a value to initialize the volume
    // with. If `fill` is missing, fills the Volume with random numbers.
    public convenience init(width: Int, height: Int, depth: Int, fill: Float? = nil) {
        self.init()
        self.width = width
        self.height = height
        self.depth = depth
        
        let n = width*height*depth
        if let fill = fill {
            values = [Float](repeating: fill, count: n)
        } else {
            // weight normalization is done to equalize the output
            // variance of every neuron, otherwise neurons with a lot
            // of incoming connections have outputs of larger variance
            let scale = sqrt(1.0/Float(width*height*depth))
            values = (0 ..< n).map{ _ in Random.Normal.float(mean: 0.0, standardDeviation: scale) }
        }
        gradients = ArrayUtils.zerosFloat(n)
    }
    
    public func get(x: Int, y: Int, d: Int) -> Float {
        let ix=((width * y)+x)*depth+d
        return values[ix]
    }
    
    public func set(x: Int, y: Int, d: Int, v: Float) {
        let ix=((width * y)+x)*depth+d
        values[ix] = v
    }
    
    public func add(x: Int, y: Int, d: Int, v: Float) {
        let ix=((width * y)+x)*depth+d
        values[ix] += v
    }
    
    public func getGrad(x: Int, y: Int, d: Int) -> Float {
        let ix = ((width * y)+x)*depth+d
        return gradients[ix]
    }
    
    public func setGrad(x: Int, y: Int, d: Int, v: Float) {
        let ix = ((width * y)+x)*depth+d
        gradients[ix] = v
    }
    
    public func addGrad(x: Int, y: Int, d: Int, v: Float) {
        let ix = ((width * y)+x)*depth+d
        gradients[ix] += v
    }
    
    public func cloneAndZero() -> Volume {
        return Volume(width: width, height: height, depth: depth, fill: 0.0)
    }
    
    public func clone() -> Volume {
        let volume = Volume(width: width, height: height, depth: depth, fill: 0.0)
        volume.values = values
        return volume
    }
    
    public func addFrom(_ volume: Volume) {
        assert(values.count == volume.values.count)
        values = zip(values, volume.values).map(+)
    }
    
    public func addFromScaled(_ volume: Volume, a: Float) {
        assert(values.count == volume.values.count)
        values = zip(values, volume.values.map{a*$0}).map(+)
    }
    
    public func setConst(_ fill: Float) {
        values = [Float](repeating: fill, count: values.count)
    }
    
    public func description() -> String {
        return "size: \(width)*\(height)*\(depth)\nw:\n\(values)\ndw:\n\(gradients)"
    }
    
    public func debugDescription() -> String {
        return "size: \(width)*\(height)*\(depth)\nw:\n\(values)\ndw:\n\(gradients)"
    }
}


