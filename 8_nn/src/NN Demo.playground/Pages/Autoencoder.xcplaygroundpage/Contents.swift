let input = InputLayerOpt(outSx: 32, outSy: 32, outDepth: 1)
let fc1 = FullyConnectedLayerOpt(numNeurons: 50, activation: .tanh)
let fc2 = FullyConnectedLayerOpt(numNeurons: 50, activation: .tanh)
let fc3 = FullyConnectedLayerOpt(numNeurons: 2)
let fc4 = FullyConnectedLayerOpt(numNeurons: 50, activation: .tanh)
let fc5 = FullyConnectedLayerOpt(numNeurons: 50, activation: .tanh)
        
let regression = RegressionLayerOpt(numNeurons: 32*32)

let net = Net([input, fc1, fc2, fc3, fc4, fc5, regression])
        
var solverOpts = SolverOpt()
solverOpts.learningRate = 1
solverOpts.method = .adadelta
solverOpts.batchSize = 50
solverOpts.l2Decay = 0.001
solverOpts.l1Decay = 0.001

let solver = Solver(net: net, options: solverOpts)
let image = UIImage(named: "Nüra_sq", in: Bundle(for: AutoencoderTest.self), compatibleWith: nil)!
var v = image.toVolume()!
let res = solver.train(x: &v, y: v.w)
print(res)
