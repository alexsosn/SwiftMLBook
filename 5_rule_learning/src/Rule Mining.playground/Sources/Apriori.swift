//
//  Apriori.swift
//  RuleMining
//
//  Created by Alex Sosnovshchenko on 3/31/17.
//  Copyright © 2017 Alex Sosnovshchenko. All rights reserved.
//

import Foundation

public class Apriori<I: Hashable & Equatable & CustomStringConvertible> {
    public typealias ItemSet = Set<Int>

    class Subsets: Sequence {
        var subsets: [ItemSet]
        
        init(_ set: ItemSet) {
            self.subsets = Array(set).combinations().map(Set.init)
        }
        
        func makeIterator() -> AnyIterator<ItemSet> {
            return AnyIterator { [weak self] in
                guard let `self` = self else {
                    return nil
                }
                return self.subsets.popLast()
            }
        }
    }
    
    public struct Rule: CustomStringConvertible {
        let ifPart: Set<I>
        let thenPart: Set<I>
        
        public var description: String {
            
            let ifDescription = ifPart.reduce("") {
                $0.characters.count == 0 ? $1.description : $0+", "+$1.description
            }
            
            let thenDescription = thenPart.reduce("") {
                $0.characters.count == 0 ? $1.description : $0+", "+$1.description
            }
            
            return "{ \(ifDescription) → \(thenDescription) }"
        }
    }
    
    public var elements: Set<Int>
    public let transactions: ContiguousArray<ItemSet>
    public let map: [I: Int]
    public let invertedMap: [Int: I]

    // Supports are stored here to prevent multiple computations.
    internal var supports = [ItemSet: Double]()
    let total: Double
    
    public convenience init(transactions: [[I]]) {
        self.init(transactions: transactions.map(Set<I>.init))
    }
    
    public init(transactions: [Set<I>]) {
        // delete

        var indexedTransactions = [ItemSet]()
        var counter = 0
        var map = [I: Int]()
        var invertedMap = [Int: I]()

        for transaction in transactions {
            var indexedTransaction = ItemSet()
            for item in transaction {
                if let stored = map[item] {
                    indexedTransaction.insert(stored)
                } else {
                    map[item] = counter
                    invertedMap[counter] = item
                    indexedTransaction.insert(counter)
                    counter += 1
                }
            }
            indexedTransactions.append(indexedTransaction)
        }
        
        self.transactions = ContiguousArray(indexedTransactions)
        self.elements = self.transactions.reduce(Set<Int>()) {$0.union($1)}
        self.map = map
        self.invertedMap = invertedMap

        self.total = Double(self.transactions.count)
    }
    
    public func associationRules(minSupport: Double, minConfidence: Double) -> [Rule] {
        var rules = [Rule]()
        let frequent = frequentItemSets(minSupport: minSupport)
        
        for itemSet in frequent {
            for (ifPart, thenPart) in nonOverlappingSubsetPairs(itemSet) {
                if confidence(ifPart, thenPart) >= minConfidence {
                    let rule = Rule(ifPart: convertIndexesToItems(ifPart), thenPart: convertIndexesToItems(thenPart))
                    rules.append(rule)
                }
            }
        }
        
        return rules
    }
    
    
    // todo: remove this conversion
    public func convertIndexesToItems(_ itemSet: ItemSet) -> Set<I> {
        return Set(itemSet.flatMap{ self.invertedMap[$0] })
    }
    
    public func convertItemsToIndexes(_ items: Set<I>) -> ItemSet {
        return Set(items.flatMap{ self.map[$0] })
    }

    
    // Returns a set of all item sets with the support larger than minSupport.
    func frequentItemSets(minSupport: Double) -> Set<ItemSet> {
        
        var itemSets = Set<ItemSet>()
        
        let emptyItemSet: ItemSet = ItemSet()
        
        supporters[emptyItemSet] = Array(0 ..< transactions.count)

        var queue = PriorityQueue<ItemSet>(order: { (lh, rh) -> Bool in
            lh.count > rh.count
        }, startingValues: [emptyItemSet])
        
        while let itemset = queue.pop() {
            var isMax = true
            
            // Note that algorithm is generating the same candidates multiple times here. This is one of the downsides of Apriori.
            
            for anExtension in allExtensions(itemset) {
                if isAboveSupportThreshold(anExtension, extending: itemset, threshold: minSupport) {
                    isMax = false
                    queue.push(anExtension)
                }
            }
            if isMax == true {
                itemSets.insert(itemset)
            }
        }
        
        return itemSets
    }
    
    func allExtensions(_ set: ItemSet) -> LazyRandomAccessCollection<[ItemSet]> {
        let elementsMissedFromSet = elements.subtracting(set)
        let extensions = elementsMissedFromSet.map{ set.union(ItemSet([$0])) }
        return extensions.lazy
    }
    
    func nonOverlappingSubsetPairs(_ itemSet: ItemSet) -> [(ItemSet, ItemSet)] {
        var result = [(ItemSet, ItemSet)]()
        let ifParts = Subsets(itemSet)
        for ifPart in ifParts {
            let nonOverlapping = itemSet.subtracting(ifPart)
            let thenParts = Subsets(nonOverlapping)
            for thenPart in thenParts {
                result.append((ifPart, thenPart))
            }
        }
        return result
    }
    
    var supporters = [ItemSet: [Int]]()
}
