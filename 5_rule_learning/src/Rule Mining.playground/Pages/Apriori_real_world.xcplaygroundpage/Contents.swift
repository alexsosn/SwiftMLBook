//: [Previous](@previous)
//: # Apriori real-world example
import Foundation

let transactions = [["Grapes", "Cheese"],
                    ["Cheese", "Milk"],
                    ["Apples", "Oranges", "Cheese", "Gingerbread", "Marshmallows", "Eggs", "Canned vegetables"],
                    ["Tea", "Apples", "Bagels", "Marshmallows", "Ice cream", "Canned vegetables"],
                    ["Cheese", "Buckwheat", "Cookies", "Oatmeal", "Bananas", "Butter", "Bread", "Apples", "Baby puree"],
                    ["Baby puree", "Cookies"],
                    ["Cookies"],
                    ["Chicken", "Grapes", "Pizza", "Cheese", "Marshmallows", "Cream"],
                    ["Potatoes"],
                    ["Chicken"],
                    ["Corn flakes", "Cookies", "Oatmeal"],
                    ["Tea"],
                    ["Chicken"],
                    ["Chicken", "Eggs", "Cheese", "Oatmeal", "Bell pepper", "Bread", "Chocolate butter", "Buckwheat", "Tea", "Rice", "Corn", "Corn flakes", "Juice", "Sugar"],
                    ["Bread", "Canned vegetables"],
                    ["Carrot", "Beetroot", "Apples", "Sugar", "Buckwheat", "Rice", "Pasta", "Salt", "Rice flour", "Dates", "Tea", "Butter", "Beef", "Cheese", "Eggs", "Bread", "Cookies"],
                    ["Apples", "Cheese", "Chicken", "Eggs", "Corn flakes", "Cookies"],
                    ["Chicken", "Cheese", "Cookies", "Bread", "Butter"],
                    ["Cheese", "Corn flakes"],
                    ["Candies", "Cinnamon"],
                    ["Ice cream"],
                    ["Corn flakes", "Juice", "Cookies", "Apples"],
                    ["Baby puree"],
                    ["Cookies", "Cheese", "Bread", "Baby puree", "Coffee", "Chicken", "Canned vegetables", "Mango", "Eggs", "Butter"],
                    ["Bananas", "Buckwheat", "Persimmon", "Chicken", "Apples", "Butter", "Corn flakes"],
                    ["Corn flakes"],
                    ["Mango", "Corn flakes", "Baby puree", "Bread"],
                    ["Sugar", "Chicken", "Cheese", "Canned vegetables", "Apples", "Dried apricots", "Dried cranberry", "Dried pineapple", "Cookies", "Cookies", "Prune", "Oatmeal", "Tea", "Buckwheat", "Persimmon", "Beetroot"]
]

let apriori = Apriori<String>(transactions: transactions)
let rules = apriori.associationRules(minSupport: 0.15, minConfidence: 0.5)
for rule in rules {
    print(rule)
    print("Confidence: ", apriori.confidence(rule), "Lift: ", apriori.lift(rule), "Conviction: ", apriori.conviction(rule))
}
