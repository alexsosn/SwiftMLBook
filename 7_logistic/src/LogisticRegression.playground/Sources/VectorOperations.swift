//
//  VectorOperations.swift
//  Pine
//
//  Created by Oleksandr on 6/24/17.
//  Copyright © 2017 OWL. All rights reserved.
//

import Foundation
import Accelerate

// Add two vectors. Equivalent to zip(a, b).map(+)
func vecAdd(_ a: [Double], _ b: [Double]) -> [Double] {
    let count = a.count
    assert(count == b.count, "Vectors must be of equal length.")
    var c = [Double](repeating: 0.0, count: count)
    vDSP_vaddD(a, 1, b, 1, &c, 1, vDSP_Length(count))
    return c
}

// Subtract vector b from vector a. Equivalent to zip(a, b).map(-)
func vecSubtract(_ a: [Double], _ b: [Double]) -> [Double] {
    let count = a.count
    assert(count == b.count, "Vectors must be of equal length.")
    var c = [Double](repeating: 0.0, count: count)
    vDSP_vsubD(b, 1, a, 1, &c, 1, vDSP_Length(count))
    return c
}

// Multiply two vectors elementwise. Equivalent to zip(a, b).map(*)
func vecMultiply(_ a: [Double], _ b: [Double]) -> [Double] {
    let count = a.count
    assert(count == b.count, "Vectors must be of equal length.")
    var c = [Double](repeating: 0.0, count: count)
    vDSP_vmulD(a, 1, b, 1, &c, 1, vDSP_Length(count))
    return c
}

// Divide vector a by vector b elementwise. Equivalent to zip(a, b).map(/)
func vecDivide(_ a: [Double], _ b: [Double]) -> [Double] {
    let count = a.count
    assert(count == b.count, "Vectors must be of equal length.")
    var c = [Double](repeating: 0.0, count: count)
    // Note that parameters a and b are swapped.
    vDSP_vdivD(b, 1, a, 1, &c, 1, vDSP_Length(count))
    return c
}

func vecNormalize(vec: [Double]) -> (normalizedVec: [Double], mean: Double, std: Double) {
    let count = vec.count
    var mean = 0.0
    var std = 0.0
    var normalizedVec = [Double](repeating: 0.0, count: count)
    vDSP_normalizeD(vec, 1, &normalizedVec, 1, &mean, &std, vDSP_Length(count))
    return (normalizedVec, mean, std)
}
