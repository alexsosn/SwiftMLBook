import Foundation


// Array utilities

public struct ArrayUtils {
    static func zerosInt(_ n: Int) -> [Int] {
        return [Int](repeating: 0, count: n)
    }
    
    static func zerosFloat(_ n: Int) -> [Float] {
        return [Float](repeating: 0.0, count: n)
    }
    
    static func zerosBool(_ n: Int) -> [Bool] {
        return [Bool](repeating: false, count: n)
    }
    
    static func arrUnique(_ arr: [Int]) -> [Int] {
        return Array(Set(arr))
    }
}

// return max and min of a given non-empty array.
public struct Maxmin {
    var maxi: Int
    var maxv: Float
    var mini: Int
    var minv: Float
    var dv: Float
}

func maxmin(_ w: [Float]) -> Maxmin? {
    guard (w.count > 0),
        let maxv = w.max(),
        let maxi = w.index(of: maxv),
        let minv = w.min(),
        let mini = w.index(of: minv)
        else {
            return nil
    }
    return Maxmin(maxi: maxi, maxv: maxv, mini: mini, minv: minv, dv: maxv-minv)
}


