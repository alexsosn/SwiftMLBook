//: # Association measures
import Foundation
let transactions = [["🌭", "🍅", "☕", "🍪"],
                    ["🌭", "🍅", "🍪"],
                    ["🌭", "🍅", "☕"],
                    ["🌭", "🍅"],
                    ["☕", "🍪"],
                    ["☕", "🍪"],
                    ["🌭"]
]

var apriori = Apriori<String>(transactions: transactions)

//: Support for the complete itemset
let itemSet = apriori.convertItemsToIndexes(Set(["🌭", "🍅", "☕", "🍪"]))
apriori.support(itemSet)

//: Supports for {🍅} and {🌭}
let tomato = apriori.convertItemsToIndexes(Set(["🍅"]))
let hotdog = apriori.convertItemsToIndexes(Set(["🌭"]))

apriori.support(tomato)
apriori.support(hotdog)

//: Association measures for {🍅 → 🌭}
apriori.confidence(tomato, hotdog)
apriori.lift(tomato, hotdog)
apriori.conviction(tomato, hotdog)

//: Association measures for {🌭 → 🍅}
apriori.confidence(hotdog, tomato)
apriori.lift(hotdog, tomato)
apriori.conviction(hotdog, tomato)



//: [Next](@next)
