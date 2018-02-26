//: simd - noun: single instruction, multiple data.

import simd

let firstVector = float4(1.0, 2.0, 3.0, 4.0)
let secondVector = float4(5.0) // a vector filled with 5s
let dotProduct = dot(firstVector, secondVector)


//: # XOR neural network
//: ![](xor.png)
extension Float {
    init(_ bool: Bool) {
        self =  bool ? 1.0 : 0.0
    }
}

func xor(_ a: Bool, _ b: Bool) -> Bool {
    let input = float2(Float(a), Float(b))

    let weights1 = float2(1.0, 1.0)
    let weights2 = float2(-1.0, -1.0)
    
    let matrixOfWeights1 = float2x2([weights1, weights2])
    let weightedSums = input * matrixOfWeights1

    let stepLayer = float2(0.5, -1.5)
    
    let secondLayerOutput = step(weightedSums, edge: stepLayer)
    
    let weights3 = float2(1.0, 1.0)
    let outputStep: Float = 1.5
    
    let weightedSum3 = reduce_add(secondLayerOutput * weights3)
    
    let result = weightedSum3 > outputStep
    return result
}

precondition(xor(true, true) == false)
precondition(xor(false, true) == true)
precondition(xor(true, false) == true)
precondition(xor(false, false) == false)
