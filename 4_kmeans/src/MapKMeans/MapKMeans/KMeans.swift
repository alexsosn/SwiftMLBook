
import Foundation

public struct KMeans {
    public enum InitializationMethod {
        case random
        case plusplus
    }
    
    // Number of clusters.
    public let k: Int
    
    // Standard k-means algorithm designed to be used only with the Euclidean distance.
    internal let distanceMetric = Euclidean.distance
    
    internal var data: [[Double]] = []
    public var centroids: [[Double]] = []
    private(set) var clusters: [Int] = []
    // Within-cluster sum of squares
    public var WCSS: Double = 0.0
    
    public init (k: Int, initialization: InitializationMethod = .plusplus) {
        self.k = k
    }
    
    public var initialization: InitializationMethod = .plusplus
    
    internal mutating func chooseCentroidsAtRandom() {
        let uniformWeights = [Double](repeating: 1.0 , count: data.count)
        let randomIndexesNoReplacement = Random.Weighted.indicesNoReplace(weights:uniformWeights, count: k)
        
        var centroidID = 0
        for index in randomIndexesNoReplacement {
            centroids.append(data[index])
            clusters[index] = centroidID
            centroidID += 1
        }
    }
    
    // The "++" part of k-means++ algorithm.
    internal mutating func chooseCentroids() {
        let n = data.count
        
        var minDistances = [Double](repeating: Double.infinity, count: n)
        var centerIndices = [Int]()
        
        // clusterID is an integer identifier of a cluster: first cluster has identifier 0, second - 1 and so on.
        for clusterID in 0 ..< k {
            
            var pointIndex: Int
            if clusterID == 0 {
                // Choose first center randomly from data points
                pointIndex = Random.Uniform.int(n)
            } else {
                // In all other cases, choose center from the weighted distribution, proportionally to the squared 
                // distance to the closest center.
                if let nextCenter = Random.Weighted.indicesNoReplace(weights: minDistances, count: 1).first {
                    pointIndex = nextCenter
                } else {
                    fatalError()
                }
            }
            
            // Select this point to be a next center.
            centerIndices.append(pointIndex)
            let center = data[pointIndex]
            centroids.append(center)
            
            // Distance to the closest center is 0. Hence, probability of sampling once again is also 0.
            minDistances[pointIndex] = 0.0
            clusters[pointIndex] = clusterID
            
            // Calculate distance from each of data points to the center.
            
            var nextI = (0, centerIndices.first ?? Int.max)
            
            for (pointIndex, point) in data.enumerated() {
                
                // Skip the data point if it was selected as a center already.
                if pointIndex == nextI.1 {
                    // Check if all centroids attended.
                    if nextI.0 < clusterID {
                        let nextIndex = nextI.0+1
                        nextI = (nextIndex, centerIndices[nextIndex])
                    }
                    continue
                }
                // If data point is not selected as a center yet, calculate the distance from it to the last selected
                // center.
                let distance = pow(distanceMetric(point, center), 2)
                
                // Remember newly calculated distance if it is less, then minimum distance saved for the corresponding
                // data point previously.
                let currentMin = minDistances[pointIndex]
                if currentMin > distance {
                    minDistances[pointIndex] = distance
                    clusters[pointIndex] = clusterID
                }
            }
        }
    }
    
    /// Count of data points should be greater or equal to k.
    @discardableResult
    public mutating func train(data: [[Double]]) -> [Int] {
        let n = data.count
        precondition(k <= n)
        precondition(n > 0)
        
        let d = data[0].count
        precondition(d > 0)

        self.data = data

        // The number of clusters is equal to number of data points. Create a cluster for each data point.
        if k == n {
            centroids = data
            clusters = Array<Int>(0..<k)
            return clusters
        }
        
        clusters = [Int](repeating: 0, count: n)
        
        // 0. Initialize centroids
        switch initialization {
        case .random:
            chooseCentroidsAtRandom()
        case .plusplus:
            chooseCentroids()
        }
        
        while true {
            WCSS = 0.0
            
            // 2. Update Step
            // Iterate over data points and update information about clusters assignment according to the distance to
            // the closest centroid.
            for (pointIndex, point) in data.enumerated() {
                
                var minDistance = Double.infinity
                for (clusterID, centroid) in centroids.enumerated() {
                    
                    let distance = pow(distanceMetric(point, centroid), 2)
                    
                    // Remember newly calculated distance if it is less, then minimum distance saved for
                    // the corresponding data point previously.
                    if minDistance > distance {
                        clusters[pointIndex] = clusterID
                        minDistance = distance
                    }
                }
                // Save information about within-cluster squared distance for the elbow method.
                WCSS += minDistance
            }
            
            // 1. Assignment step
            // Calculate the centroids of clusters.
            var centroidsCount = [Double](repeating: 0.0, count: k)
            let rowStub = [Double](repeating: 0.0, count: d)
            var centroidsCumulative = [[Double]](repeating: rowStub, count: k)
            
            for (point, clusterID) in zip(data, clusters) {
                centroidsCount[clusterID] += 1
                centroidsCumulative[clusterID] = vecAdd(centroidsCumulative[clusterID], point)
            }
            
            var newCentroids = centroidsCumulative
            for (j, row) in centroidsCumulative.enumerated() {
                for (i, element) in row.enumerated() {
                    let newCentroid = element/centroidsCount[j]
                    assert(!newCentroid.isNaN)
                    newCentroids[j][i] = newCentroid
                }
            }
            
            // Check convergence conditions.
            var convergence = false
            convergence = zip(centroids, newCentroids).map{$0.0 == $0.1}.reduce(true, and)
            if convergence { break }
            centroids = newCentroids
        }
        
        return clusters
    }
}
