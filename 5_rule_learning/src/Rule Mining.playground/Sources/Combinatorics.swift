//
//  CombinatorialUtilities.swift
//  RuleMining
//
//  Created by Alex Sosnovshchenko on 3/31/17.
//  Copyright © 2017 Alex Sosnovshchenko. All rights reserved.
//

import Foundation

public extension Array {
    
    // Combinations are found via the binary representation of subsets.
    // See for example: https://math.stackexchange.com/a/240770
    public func combinations() -> [[Element]] {
        if isEmpty { return [] }
        let numberOfSubsets = Int(pow(2, Double(count)))
        var result = [[Element]]()
        for i in 1..<numberOfSubsets {
            var remainder = i
            var index = 0
            var combination = [Element]()
            while remainder > 0 {
                if remainder % 2 == 1 {
                    combination.append(self[index])
                }
                index += 1
                remainder /= 2
            }
            result.append(combination)
        }
        return result
    }
}
