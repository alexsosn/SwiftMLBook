//
//  Euclidean.swift
//  
//
//  Created by Alex Sosnovshchenko on 4/8/17.
//  Copyright © 2017 Alex Sosnovshchenko. All rights reserved.
//

import Foundation
import Accelerate

public struct Euclidean {
    public static func distance(_ lhVec: [Double], _ rhVec: [Double]) -> Double {
        precondition(lhVec.count == rhVec.count)
        var dist = 0.0
        vDSP_distancesqD(UnsafePointer(lhVec), 1, UnsafePointer(rhVec), 1, &dist, vDSP_Length(lhVec.count))
        return sqrt(dist)
    }
}

