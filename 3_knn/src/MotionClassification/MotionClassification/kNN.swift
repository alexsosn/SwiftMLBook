
import Foundation

public struct kNN<X, Y> where Y: Hashable {
    public var k: Int
    public let distanceMetric: (_ x1: X, _ x2: X) -> Double
    
    private var data: [(X, Y)] = []
    
    public init (k: Int, distanceMetric: @escaping (_ x1: X, _ x2: X) -> Double) {
        assert(k > 0, "Error, k must be greater then 0.")
        self.k = k
        self.distanceMetric = distanceMetric
    }
    
    public mutating func train(X: [X], y: [Y]) {
        data.append(contentsOf: zip(X, y))
    }
    
    public func predict(x: X) -> Y? {
        assert(data.count > 0, "Please, use method train() at first to provide training data.")
        
        let tuples = data
            .map { (distanceMetric(x, $0.0), $0.1) }    // calculate tuples (distance, label)
            .sorted { $0.0 < $1.0 }                     // sort descending by distances
            .prefix(upTo: k)                            // take first k elements
        
        let countedSet = NSCountedSet(array: tuples.map{$0.1})
        
        // sort ascending by frequency
        let result = countedSet.allObjects.sorted {
            countedSet.count(for: $0) > countedSet.count(for: $1)
            }.first
        
        return result as? Y
    }
}
