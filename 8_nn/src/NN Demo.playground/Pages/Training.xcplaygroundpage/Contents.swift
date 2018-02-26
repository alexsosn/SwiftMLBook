srand48(time(nil))


let input = InputLayerOpt(outSx: 1, outSy: 1, outDepth: 2)
let fc1 = FullyConnectedLayerOpt(numNeurons: 50, activation: .tanh)
let fc2 = FullyConnectedLayerOpt(numNeurons: 40, activation: .tanh)
let softmax = SoftmaxLayerOpt(numClasses: 3)

net = Net([input, fc1, fc2, softmax])

var solverOpts = SolverOpt()
solverOpts.learningRate = 0.0001
solverOpts.momentum = 0.0
solverOpts.batchSize = 1
solverOpts.l2Decay = 0.0
solver = Solver(net: net!, options: solverOpts)


// tanh are their own layers. Softmax gets its own fully connected layer.
// this should all get desugared just fine.
assert(net!.layers.count == 7)

// should forward prop volumes to probabilities
var x = Volume(array: [0.2, -0.3])
let probabilityVolume = net!.forward(&x)

assert(probabilityVolume.w.count == 3)  // 3 classes output
var w = probabilityVolume.w
for i in 0 ..< 3 {
    XCTAssertGreaterThan(w[i], 0.0)
    XCTAssertLessThan(w[i], 1.0)
}

assert(abs(w[0]+w[1]+w[2] - 1.0) < 0.000000000001)

// should increase probabilities for ground truth class when trained
// lets test 100 random point and label settings
// note that this should work since l2 and l1 regularization are off
// an issue is that if step size is too high, this could technically fail...
for _ in 0 ..< 100 {
    var x = Volume(array: [Random.Uniform.double() * 2 - 1, Random.Uniform.double() * 2 - 1])
    let pv = net!.forward(&x)
    let gti = Int(Random.Uniform.double() * 3)
    let trainRes = solver!.train(x: &x, y: gti)
    print(trainRes)
    
    let pv2 = net!.forward(&x)
    assert(pv2.w[gti] > pv.w[gti])
}

// should compute correct gradient at data
// here we only test the gradient at data, but if this is
// right then that's comforting, because it is a function
// of all gradients above, for all layers.

var x = Volume(array: [Random.Uniform.double() * 2.0 - 1.0, Random.Uniform.double() * 2.0 - 1.0])
let gti = Int(Random.Uniform.double() * 3) // ground truth index
let res = solver!.train(x: &x, y: gti) // computes gradients at all layers, and at x

print(res)

let Δ = 0.000001

for i: Int in 0 ..< x.w.count {
    
    // finite difference approximation
    
    let gradAnalytic = x.dw[i]
    
    let xold = x.w[i]
    x.w[i] += Δ
    let c0 = net!.getCostLoss(V: &x, y: gti)
    x.w[i] -= 2*Δ
    let c1 = net!.getCostLoss(V: &x, y: gti)
    x.w[i] = xold // reset
    
    let gradNumeric = (c0 - c1)/(2.0 * Δ)
    let relError = abs(gradAnalytic - gradNumeric)/abs(gradAnalytic + gradNumeric)
    print("\(i): numeric: \(gradNumeric), analytic: \(gradAnalytic) => rel error \(relError)")
    
    assert(relError < 1e-2)
    
}
