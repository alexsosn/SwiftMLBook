//: [Previous](@previous)

import Foundation

//: Inverse matrix
// Calculate inverse matrix.
// Pass flattened matrix in row-major order.

let aMat: [Double] = [-1, 3/2, 1, -1]

let inv = inverse(aMat, dimension: 2)

//: GEMM
//: C←αAB + βC

let result = gemm(aMat: xMatFlattened, bMat: diffVec, rowsAC: featureCount, colsBC: labelSize, colsA_rowsB: sampleCount, transposeA: true, α: scaleBy)


//: Normalize matrix
let (normalizedXMat, xMeanVec, xStdVec) = matNormalize(matrix: flattenedXMat, rows: sampleCount, columns: featureCount)

//: [Next](@next)
