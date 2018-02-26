
import Foundation
import CoreGraphics
// http://stackoverflow.com/questions/25050309/swift-random-float-between-0-and-1

public extension ClosedRange where Bound : FloatingPoint {
    public func random() -> Bound {
        let range = self.upperBound - self.lowerBound
        let randomValue = (Bound(arc4random_uniform(UINT32_MAX)) / Bound(UINT32_MAX)) * range + self.lowerBound
        return randomValue
    }
}

extension Collection {
    public func random() -> Self.Element {
        if let startIndex = self.startIndex as? Int {
            let start = UInt32(startIndex)
            let end = UInt32(self.endIndex as! Int)
            return self[Int(arc4random_uniform(end - start) + start) as! Self.Index]
        }
        var generator = self.makeIterator()
        var count = arc4random_uniform(UInt32(self.count as! Int))
        while count > 0 {
            _ = generator.next()
            count = count - 1
        }
        return generator.next()!
    }
}

/// Random number utilities
public struct Random {
    
    public struct Uniform {
        private static let seed: Void = srand48(Int(Date().timeIntervalSince1970))
        
        // should be [0 .. 1)
        public static func double() -> Double {
            _ = seed
            return drand48()
        }
        
        // should be [0 .. 1)
        func cgFloat() -> CGFloat {
            return CGFloat(Float(arc4random()) / Float(UINT32_MAX))
        }
        
        public static func float() -> Float {
            _ = seed
            return Float(drand48())
        }
        
        // Less then upper bound: [0 .. upperBound)
        public static func int(_ upperBound: Int) -> Int {
            return Int(arc4random_uniform(UInt32(upperBound)))
        }
        
        public static func double(_ a: Double, _ b: Double) -> Double {
            return double()*(b-a)+a
        }

        public static func float(_ a: Float, _ b: Float) -> Float {
            return float()*(b-a)+a
        }
        
        public static func int(_ a: Int, _ b: Int) -> Int {
            return Int(floor(double()))*(b-a)+a
        }
    }
    
    public struct Normal {
        public static func double(mean: Double, standardDeviation: Double) -> Double {
            let u1 = Uniform.double()//Double(arc4random()) / Double(UINT32_MAX)
            let u2 = Uniform.double()//Double(arc4random()) / Double(UINT32_MAX)

            let r2 = -2.0 * log(u1)
            let theta = 2.0 * Double.pi * u2
            
            return standardDeviation * (sqrt(r2) * cos(theta)) + mean
        }
        
        public static func float(mean: Float, standardDeviation: Float) -> Float {
            let u1 = Uniform.float()//Double(arc4random()) / Double(UINT32_MAX)
            let u2 = Uniform.float()//Double(arc4random()) / Double(UINT32_MAX)
            
            let r2 = -2.0 * log(u1)
            let theta = 2.0 * Float.pi * u2
            
            return standardDeviation * (sqrt(r2) * cos(theta)) + mean
        }
//        // Box-Muller Transformation
//        // http://mathworld.wolfram.com/Box-MullerTransformation.html
//        public static func double(mean: Double, std: Double) -> Double {
//            let u1 = Double(arc4random()) / Double.greatestFiniteMagnitude // uniform distribution
//            let u2 = Double(arc4random()) / Double.greatestFiniteMagnitude // uniform distribution
//            let f1 = sqrt(-2 * log(u1))
//            let f2 = 2 * Double.pi * u2
//            let g1 = f1 * cos(f2) // gaussian distribution
////            let g2 = f1 * sin(f2) // gaussian distribution
//
//            return (g1 + mean)*std
//        }
//
//        public static func float(mean: Float, std: Float) -> Float {
//            let u1 = Float(arc4random()) / Float.greatestFiniteMagnitude // uniform distribution
//            let u2 = Float(arc4random()) / Float.greatestFiniteMagnitude // uniform distribution
//            let f1 = sqrt(-2 * log(u1))
//            let f2 = 2 * Float.pi * u2
//            let g1 = f1 * cos(f2) // gaussian distribution
////            let g2 = f1 * sin(f2) // gaussian distribution
//
//            return (g1 + mean)*std
//        }
    }
}
