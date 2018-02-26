//
//  MatrixOperations.swift
//  Pine
//
//  Created by Oleksandr on 6/25/17.
//  Copyright © 2017 OWL. All rights reserved.
//

import Foundation
import Accelerate

enum MatrixOperationsErrors: Error {
    case notInvertible
}

// Calculate inverse matrix.
// Pass flattened matrix in row-major order.
func inverse(_ matrix: [Double], dimension: Int) throws -> [Double] {
    var results = matrix
    
    var integerPivot = [__CLPK_integer](repeating: 0, count: dimension * dimension)
    var workspaceLength = __CLPK_integer(dimension * dimension)
    var workspace = [CDouble](repeating: 0, count: Int(workspaceLength))
    var error: __CLPK_integer = 0
    var rowsVar1 = __CLPK_integer(dimension)
    var rowsVar2 = rowsVar1
    var rowsVar3 = rowsVar1
    
    dgetrf_(&rowsVar1, &rowsVar2, &results, &rowsVar3, &integerPivot, &error)
    // fixme: error here
    if error != 0 {
        throw MatrixOperationsErrors.notInvertible
    }
    
    dgetri_(&rowsVar1, &results, &rowsVar2, &integerPivot, &workspace, &workspaceLength, &error)
    if error != 0 {
        throw MatrixOperationsErrors.notInvertible
    }
    
    return results
}

// C←αAB + βC
// Pass flattened matrices in row-major order.
// rowsAC, colsBC, colsA_rowsB - Count of rows/columns AFTER transpose.
func gemm(aMat: [Double], bMat: [Double], cMat: [Double]? = nil,
          rowsAC: Int, colsBC: Int, colsA_rowsB: Int,
          transposeA: Bool = false, transposeB: Bool = false,
          α: Double = 1, β: Double = 0) -> [Double] {
    var result = cMat ?? [Double](repeating: 0.0, count: rowsAC*colsBC)
    
    // C←αAB + βC
    cblas_dgemm(CblasRowMajor, // Specifies row-major (C) or column-major (Fortran) data ordering.
        transposeA ? CblasTrans : CblasNoTrans, // Specifies whether to transpose matrix A.
        transposeB ? CblasTrans : CblasNoTrans, // Specifies whether to transpose matrix B.
        Int32(rowsAC), // Number of rows in matrices A and C.
        Int32(colsBC), // Number of columns in matrices B and C.
        Int32(colsA_rowsB), // Number of columns in matrix A; number of rows in matrix B.
        α, // α.
        aMat, // Matrix A.
        transposeA ? Int32(rowsAC) : Int32(colsA_rowsB), // The size of the first dimention of matrix A; if you are passing a matrix A[m][n], the value should be m.
        bMat, // Matrix B.
        transposeB ? Int32(colsA_rowsB) : Int32(colsBC), // The size of the first dimention of matrix B; if you are passing a matrix B[m][n], the value should be m.
        β, // β.
        &result, // Matrix C.
        Int32(colsBC) // The size of the first dimention of matrix C; if you are passing a matrix C[m][n], the value should be m.
    )
    return result
}

// Calculates mean for every matrix column.
func meanColumns(matrix: [Double], rows: Int, columns: Int) -> [Double] {
    assert(matrix.count == rows*columns)
    
    var resultVec = [Double](repeating: 0.0, count: columns)
    
    matrix.withUnsafeBufferPointer{ inputBuffer in
        resultVec.withUnsafeMutableBufferPointer{ outputBuffer in
            let inputPointer = inputBuffer.baseAddress!
            let outputPointer = outputBuffer.baseAddress!
            for i in 0 ..< columns {
                vDSP_meanvD(inputPointer.advanced(by: i), columns, outputPointer.advanced(by: i), vDSP_Length(rows))
            }
        }
    }
    return resultVec
}

// Calculates standard deviation for every matrix column.
func stdColumns(matrix: [Double], rows: Int, columns: Int) -> [Double] {
    assert(matrix.count == rows*columns)
    
    let meanVec = meanColumns(matrix: matrix, rows: rows, columns: columns)
    
    var varianceVec = [Double](repeating: 0.0, count: columns)
    var deviationsMat = [Double](repeating: 0.0, count: rows*columns)
    
    // Calculating the variance for each column.
    matrix.withUnsafeBufferPointer{ inputBuffer in
        deviationsMat.withUnsafeMutableBufferPointer{ deviationsBuffer in
            varianceVec.withUnsafeMutableBufferPointer{ outputBuffer in
                for i in 0 ..< columns {
                    let inputPointer = inputBuffer.baseAddress!.advanced(by: i)
                    let devPointer = deviationsBuffer.baseAddress!.advanced(by: i)
                    let outputPointer = outputBuffer.baseAddress!.advanced(by: i)
                    
                    var mean = -meanVec[i]
                    // Deviations of each column from its mean.
                    vDSP_vsaddD(inputPointer, columns, &mean, devPointer, columns, vDSP_Length(rows))
                    // Squared deviations.
                    vDSP_vsqD(devPointer, columns, devPointer, columns, vDSP_Length(rows))
                    // Sum for every column. Note, that parameters should be passed in a reverse order.
                    vDSP_sveD(devPointer, columns, outputPointer, vDSP_Length(rows))
                }
            }
        }
    }
    
    // -1 for Bessel's correction.
    var devideBy = Double(rows) - 1
    vDSP_vsdivD(varianceVec, 1, &devideBy, &varianceVec, 1, vDSP_Length(columns))
    
    // Calculating the standard deviation.
    var length = Int32(columns)
    var stdVec = varianceVec
    vvsqrt(&stdVec, &varianceVec, &length)
    
    return stdVec
}
// (x-μ)/σ
func matNormalize(matrix: [Double], rows: Int, columns: Int) -> (normalizedMat: [Double], meanVec: [Double], stdVec: [Double]) {
    var meanVec = meanColumns(matrix: matrix, rows: rows, columns: columns)
    var stdVec = stdColumns(matrix: matrix, rows: rows, columns: columns)
    
    var result = [Double](repeating: 0.0, count: rows*columns)
    
    matrix.withUnsafeBufferPointer{ inputBuffer in
        result.withUnsafeMutableBufferPointer{ resultBuffer in
            for i in 0 ..< columns {
                let inputPointer = inputBuffer.baseAddress!.advanced(by: i)
                let resultPointer = resultBuffer.baseAddress!.advanced(by: i)
                
                var mean = -meanVec[i]
                var std = stdVec[i]
                // Substract standard deviation.
                vDSP_vsaddD(inputPointer, columns, &mean, resultPointer, columns, vDSP_Length(rows))
                // Devide by mean.
                vDSP_vsdivD(resultPointer, columns, &std, resultPointer, columns, vDSP_Length(rows))
            }
        }
    }
    
    return (result, meanVec, stdVec)
}
