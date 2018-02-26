
import Foundation

public struct DTW {
    
    public static func distance(sVec: [Double], tVec: [Double]) -> Double {
        let n = sVec.count
        let m = tVec.count
        //rows: n+1, columns: m+1
        var dtwMat = [[Double]](repeating: [Double](repeating: Double.greatestFiniteMagnitude, count: m+1), count: n+1)
        dtwMat[0][0] = 0
        
        for i in 1...n {
            for j in 1...m {
                let cost = pow(sVec[i-1] - tVec[j-1], 2)
                let insertion = dtwMat[i-1][j]
                let deletion = dtwMat[i][j-1]
                let match = dtwMat[i-1][j-1]
                let prevMin = min(insertion, deletion, match)
                dtwMat[i][j] = cost + prevMin
            }
        }
        
        return dtwMat[n][m]/Double(max(n, m)) // normalize by sequence length
    }
    
    // With locality
    public static func distance(sVec: [Double], tVec: [Double], w: Int) -> Double {
        let n = sVec.count
        let m = tVec.count
        var dtwMat = [[Double]](repeating: [Double](repeating: Double.greatestFiniteMagnitude, count: m+1), count: n+1)
        dtwMat[0][0] = 0
        let constraint = max(w, abs(n-m))
        
        for i in 1...n {
            for j in max(1, i-constraint)...min(m, i+constraint) {
                let cost = pow(sVec[i-1] - tVec[j-1], 2)
                let insertion = dtwMat[i-1][j]
                let deletion = dtwMat[i][j-1]
                let match = dtwMat[i-1][j-1]
                dtwMat[i][j] = cost + min(insertion, deletion, match)
            }
        }
        
        return dtwMat[n][m]/Double(max(n, m)) // normalize by sequence length
    }
    
    // Convenience
    public static func distance(w: Int) -> ([Double], [Double]) -> Double {
        return {self.distance(sVec: $0, tVec: $1, w: w)}
    }
}
