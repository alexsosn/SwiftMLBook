import Foundation

///: Association measures

public extension Apriori {
    
    // Calculate support for the item set.
    // The support is calculated only for the subset of all transactions which is supporting the itemset being extended
    // by the current item set. This allows to narrow the search space and make computations faster.
    
    public func isAboveSupportThreshold(_ set: ItemSet, extending: ItemSet?, threshold: Double) -> Bool {
        
        // If current item set is extension (superset), look for supporters of subset.
        
        if let extending = extending,
            let supportersIndices = supporters[extending] {
            
            // Find transactions that support the current item set and store their indices for future extensions.
            var newSupportersIndices = [Int]()
            for i in supportersIndices {
                let transaction = transactions[i]
                
                if transaction.isSuperset(of: set) {
                    newSupportersIndices.append(i)
                }
            }
            
            let result = Double(newSupportersIndices.count)/total
            supports[set] = result
            
            if result >= threshold {
                supporters[set] = newSupportersIndices
                return true
            } else {
                return false
            }
        }
        
        return false //support(set) >= threshold
    }
    
    public func support(_ set: ItemSet) -> Double {
        // Store already calculated supports
        if let stored = supports[set] {
            return stored
        }
        
        let supporting = transactions.filter{ $0.isSuperset(of: set)}.count
        let support = Double(supporting)/total
        
        supports[set] = support
        
        return support
    }
    
    public func confidence(_ rule: Rule) -> Double {
        let thenPart = convertItemsToIndexes(rule.thenPart)
        let ifPart = convertItemsToIndexes(rule.ifPart)
        return support(ifPart.union(thenPart))/support(ifPart)
    }
    
    public func confidence(_ ifPart: ItemSet, _ thenPart: ItemSet) -> Double {
        return support(ifPart.union(thenPart))/support(ifPart)
    }
    
    public func lift(_ rule: Rule) -> Double {
        let thenPart = convertItemsToIndexes(rule.thenPart)
        return confidence(rule)/support(thenPart)
    }
    
    public func lift(_ ifPart: ItemSet, _ thenPart: ItemSet) -> Double {
        return confidence(ifPart, thenPart)/support(thenPart)
    }
    
    public func conviction(_ rule: Rule) -> Double {
        let thenPart = convertItemsToIndexes(rule.thenPart)
        return (1-support(thenPart))/(1-confidence(rule))
    }
    
    public func conviction(_ ifPart: ItemSet, _ thenPart: ItemSet) -> Double {
        return (1-support(thenPart))/(1-confidence(ifPart, thenPart))
    }
}
