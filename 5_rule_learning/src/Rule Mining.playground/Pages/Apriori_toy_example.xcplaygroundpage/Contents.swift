//: [Previous](@previous)
//: # Apriori toy example

import Foundation

let transactions = [["🌭", "🍅", "☕", "🍪"],
                    ["🌭", "🍅", "🍪"],
                    ["🌭", "🍅", "☕"],
                    ["🌭", "🍅"],
                    ["☕", "🍪"],
                    ["☕", "🍪"],
                    ["🌭"]
]

let apriori = Apriori<String>(transactions: transactions)
let rules = apriori.associationRules(minSupport: 0.3, minConfidence: 0.5)
for rule in rules {
    print(rule)
    print("Confidence: ", apriori.confidence(rule), "Lift: ", apriori.lift(rule), "Conviction: ", apriori.conviction(rule))
}

//: [Next](@next)
