
import Foundation
import GameplayKit
import UIKit

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
    
    // create random permutation of numbers, in range [0...n-1]
    public static func permutation(_ n: Int) -> [Int]{
        let dist = GKShuffledDistribution(lowestValue: 0, highestValue: n-1)
        return (0..<n).map{_ in dist.nextInt()}
    }
    
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
        
        public static func int(_ a: Int, _ b: Int) -> Int {
            return Int(floor(double()))*(b-a)+a
        }
        
        public static func randomColor(randomAlpha randomApha: Bool = false) -> UIColor {
            
            let hue = CGFloat(arc4random_uniform(1000))/1000
            let saturation = CGFloat(int(500))/1000 + 0.5
            let brightness = CGFloat(int(200))/1000 + 0.8
            let alphaValue = randomApha ? CGFloat(int(255)) / 255.0 : 1
            
            return UIColor(hue: hue, saturation: saturation, brightness: brightness, alpha: alphaValue)
        }
    }
    
    public struct Normal {
        private static let gaussDistribution = GKGaussianDistribution(randomSource: GKRandomSource(), mean:0, deviation: 1)
        
        public static func double(_ mu: Double, std: Double) -> Double {
            return (Double(gaussDistribution.nextUniform()) + mu) * std
        }
    }

    public struct Weighted {
        
        // sample from list list according to probabilities list
        // the two lists are of same size, and probabilities adds up to 1
        public static func sample(_ list: [Double], probabilities: [Double]) -> Double? {
            let p = Uniform.double(0, 1.0)
            var cumprob = 0.0
            let n = list.count
            for k in 0 ..< n {
                cumprob += probabilities[k]
                if p < cumprob { return list[k] }
            }
            return nil
        }
        
        /// Unequal probability sampling; without-replacement case
        // Ported from R sources: https://github.com/rho-devel/rho/blob/master/src/main/random.cpp
        // With a hint from here: http://stats.stackexchange.com/a/67918/77117
        // The Function returns an array of indices.
        public static func indicesNoReplace(weights: [Double], count: Int) -> [Int] {
            let n = weights.count
            precondition(count <= n)
            
            // Record element identities
            let indices = [Int](0 ... n-1)
            
            // Sort probabilities into descending order
            // Order element identities in parallel
            var p_i = zip(weights, indices).sorted { $0.0 > $1.0 }
            
            // Compute the sample
            var totalmass: Double = weights.reduce(0, +)
            
            var result = [Int]()
            
            var n1 = n-1
            for _ in 0 ..< count {
                defer { n1 -= 1 }
                let rT = totalmass * Random.Uniform.double()
                var mass = 0.0
                
                var j = 0
                for k in 0 ..< n1 {
                    j = k
                    mass += p_i[j].0
                    if rT <= mass { break }
                }
                
                result.append(p_i[j].1)
                totalmass -= p_i[j].0
                
                for k in j ..< n1 {
                    p_i[k].0 = p_i[k + 1].0
                    p_i[k].1 = p_i[k + 1].1
                }
            }
            
            return result
        }
    }
}
