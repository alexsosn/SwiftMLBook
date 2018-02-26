
import XCTest
@testable import NeuralNetwork

class NeuralNetworkTests: XCTestCase {
    
    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }
    
    func testExample() {
        // Here's a minimum example of defining a 2-layer neural network and training it on a single data point:
        
        // species a 2-layer neural network with one hidden layer of 20 neurons
        // input layer declares size of input. here: 2-D data
        // ConvNetJS works on 3-Dimensional volumes (sx, sy, depth), but if you're not dealing with images
        // then the first two dimensions (sx, sy) will always be kept at size 1
        let input = InputLayerOpt(outWidth: 1, outHeight: 1, outDepth: 2)
        // declare 20 neurons, followed by ReLU (rectified linear unit non-linearity)
        let fc = FullyConnectedLayerOpt(numNeurons: 20, activation: .relu)
        // declare the linear classifier on top of the previous hidden layer
        let softmax = SoftmaxLayerOpt(numClasses: 2)
        
        let net = Net([input, fc, softmax])
        
        // forward a random data point through the network
        var x = Volume(array: [0.3, -0.5])
        let prob = net.forward(&x)
        
        // prob is a Volume. Volumes have a field .w that stores the raw data, and .dw that stores gradients
        print("probability that x is class 0: \(prob.values[0])") // prints (for example) 0.602521202165062
        
        var traindef = SolverOpt()
        traindef.learningRate = 0.01
        traindef.l2Decay = 0.001
        
        let solver = Solver(net: net, options: traindef)
        _ = solver.train(x: &x, y: 0) // train the network, specifying that x is class zero
        
        let prob2 = net.forward(&x)
        print("probability that x is class 0: \(prob2.values[0])")
        // now prints (for example) 0.609982755733715, slightly higher than previous 0.602521202165062: the networks
        // weights have been adjusted by the Solver to give a higher probability to
        // the class we trained the network with (zero)

    }
    
    func testExample2() {
        srand48(time(nil))
        
        
        let input = InputLayerOpt(outWidth: 1, outHeight: 1, outDepth: 2)
        let fc1 = FullyConnectedLayerOpt(numNeurons: 50, activation: .tanh)
        let fc2 = FullyConnectedLayerOpt(numNeurons: 40, activation: .tanh)
        let softmax = SoftmaxLayerOpt(numClasses: 3)
        
        let net = Net([input, fc1, fc2, softmax])
        
        var solverOpts = SolverOpt()
        solverOpts.learningRate = 0.0001
        solverOpts.momentum = 0.0
        solverOpts.batchSize = 1
        solverOpts.l2Decay = 0.0
        let solver = Solver(net: net, options: solverOpts)
        
        
        // tanh are their own layers. Softmax gets its own fully connected layer.
        // this should all get desugared just fine.
        assert(net.layers.count == 7)
        
        // should forward prop volumes to probabilities
        var x = Volume(array: [0.2, -0.3])
        let probabilityVolume = net.forward(&x)
        
        assert(probabilityVolume.values.count == 3)  // 3 classes output
        var w = probabilityVolume.values
        for i in 0 ..< 3 {
            XCTAssertGreaterThan(w[i], 0.0)
            XCTAssertLessThan(w[i], 1.0)
        }
        
        assert(abs(w[0]+w[1]+w[2] - 1.0) < 0.000001)
        
        // should increase probabilities for ground truth class when trained
        // lets test 100 random point and label settings
        // note that this should work since l2 and l1 regularization are off
        // an issue is that if step size is too high, this could technically fail...
        for _ in 0 ..< 100 {
            var x = Volume(array: [Random.Uniform.float() * 2 - 1, Random.Uniform.float() * 2 - 1])
            let pv = net.forward(&x)
            let gti = Int(Random.Uniform.double() * 3)
            let trainRes = solver.train(x: &x, y: gti)
            print(trainRes)
            
            let pv2 = net.forward(&x)
            assert(pv2.values[gti] > pv.values[gti])
        }
        
        // should compute correct gradient at data
        // here we only test the gradient at data, but if this is
        // right then that's comforting, because it is a function
        // of all gradients above, for all layers.
        
        x = Volume(array: [Random.Uniform.float() * 2.0 - 1.0, Random.Uniform.float() * 2.0 - 1.0])
        let gti = Int(Random.Uniform.double() * 3) // ground truth index
        let res = solver.train(x: &x, y: gti) // computes gradients at all layers, and at x
        
        print(res)
        
        let Δ: Float = 0.000001
        
        for i in 0 ..< x.values.count {
            
            // finite difference approximation
            
            let gradAnalytic = x.gradients[i]
            
            let xold = x.values[i]
            x.values[i] += Δ
            let c0 = net.getCostLoss(volume: &x, y: gti)
            x.values[i] -= 2*Δ
            let c1 = net.getCostLoss(volume: &x, y: gti)
            x.values[i] = xold // reset
            
            let gradNumeric = (c0 - c1)/(2.0 * Δ)
            let relError = abs(gradAnalytic - gradNumeric)/abs(gradAnalytic + gradNumeric)
            print("\(i): numeric: \(gradNumeric), analytic: \(gradAnalytic) => rel error \(relError)")
            
            assert(relError < 1e-2)
        }
    }
    
    func testPerformanceExample() {
        // This is an example of a performance test case.
        self.measure {
            // Put the code you want to measure the time of here.
        }
    }
    
}
