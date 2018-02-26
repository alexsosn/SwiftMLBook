import Foundation

public enum  SolverType: String {
    case sgd
    case adam
    case adagrad
    case windowgrad
    case adadelta
    case nesterov
}

public struct SolverOpt {
    
    public init(method: SolverType = .sgd,
                batchSize: Int = 1,
                l1Decay: Float = Float(0.0),
                l2Decay: Float = Float(0.0),
                learningRate: Float = Float(0.01),
                momentum: Float = 0.9,
                ρ: Float = 0.95,
                ε: Float = 1e-8,
                β1: Float = 0.9,
                β2: Float = 0.999) {
        
        self.method = method
        self.batchSize = batchSize
        self.l1Decay = l1Decay
        self.l2Decay = l2Decay
        self.learningRate = learningRate
        self.momentum = momentum
        self.ρ = ρ
        self.ε = ε
        self.β1 = β1
        self.β2 = β2
    }
    
    public var method: SolverType
    public var batchSize: Int
    public var l1Decay: Float
    public var l2Decay: Float
    public var learningRate: Float
    public var momentum: Float
    public var ρ: Float
    public var ε: Float
    public var β1: Float
    public var β2: Float
}

public class Solver {
    
    public struct TrainResult {
        public var l2DecayLoss: Float
        public var l1DecayLoss: Float
        public var costLoss: Float
        public var softmaxLoss: Float
        public var loss: Float
    }
    
    public var net: Net
    public var learningRate: Float = 0.0
    public var l1Decay: Float = 0.0
    public var l2Decay: Float = 0.0
    public var batchSize: Int = 0
    public var method: SolverType
    public var momentum: Float = 0.0
    public var ρ: Float = 0.0
    public var ε: Float = 0.0
    public var β1: Float = 0.0
    public var β2: Float = 0.0
    public var k: Int = 0
    public var gsum: [[Float]] = []
    public var xsum: [[Float]] = []
    public var isRegression: Bool = false
    
    public init(net: Net, options: SolverOpt) {
        
        self.net = net
        
        learningRate = options.learningRate 
        l1Decay = options.l1Decay
        l2Decay = options.l2Decay
        batchSize = options.batchSize
        method = options.method // sgd/adam/adagrad/adadelta/windowgrad/netsterov
        
        momentum = options.momentum
        ρ = options.ρ // used in adadelta
        ε = options.ε // used in adam or adadelta
        β1 = options.β1 // used in adam
        β2 = options.β2 // used in adam
        
        k = 0 // iteration counter
        gsum = [] // last iteration gradients (used for momentum calculations)
        xsum = [] // used in adam or adadelta
        
        // check if regression is expected
        isRegression = net.layers[net.layers.count - 1] is RegressionLayer
    }
    
    public func train(x: inout Volume, y: [Float]) -> TrainResult {
        assert(isRegression)
        
        _ = net.forward(&x, isTraining: true) // also set the flag that lets the net know we're just training
        
        let costLoss = net.backward(y)
        let (l1DecayLoss, l2DecayLoss) = performTraining()
        
        // appending softmaxLoss for backwards compatibility, but from now on we will always use costLoss
        // in future, TODO: have to completely redo the way loss is done around the network as currently
        // loss is a bit of a hack. Ideally, user should specify arbitrary number of loss functions on any layer
        // and it should all be computed correctly and automatically.
        return TrainResult(l2DecayLoss: l2DecayLoss,
                           l1DecayLoss: l1DecayLoss,
                           costLoss: costLoss,
                           softmaxLoss: costLoss,
                           loss: costLoss + l1DecayLoss + l2DecayLoss)
    }
    
    public func train(x: inout Volume, y: (dim: Int, val: Float)) -> TrainResult {
        assert(isRegression)
        
        _ = net.forward(&x, isTraining: true) // also set the flag that lets the net know we're just training
        
        let costLoss = net.backward(y)
        let (l1DecayLoss, l2DecayLoss) = performTraining()
        
        return TrainResult(l2DecayLoss: l2DecayLoss,
                           l1DecayLoss: l1DecayLoss,
                           costLoss: costLoss,
                           softmaxLoss: costLoss,
                           loss: costLoss + l1DecayLoss + l2DecayLoss)
    }
    
    public func train(x: inout Volume, y: Int) -> TrainResult {
        assert(!isRegression, "y should be an array if you want to do a regression.")
        
        _ = net.forward(&x, isTraining: true) // also set the flag that lets the net know we're just training
        
        let costLoss = net.backward(y)
        
        let (l1DecayLoss, l2DecayLoss) = performTraining()
        
        return TrainResult(l2DecayLoss: l2DecayLoss,
                           l1DecayLoss: l1DecayLoss,
                           costLoss: costLoss,
                           softmaxLoss: costLoss,
                           loss: costLoss + l1DecayLoss + l2DecayLoss)
    }
    
    private func performTraining() -> (l1DecayLoss: Float, l2DecayLoss: Float) {
        var l2DecayLoss = Float(0.0)
        var l1DecayLoss = Float(0.0)
        
        k += 1
        if k % batchSize == 0 {
            
            var pglist = net.getParamsAndGrads()
            var newParamsAndGradients: [ParamsAndGrads] = []
            
            // initialize lists for accumulators. Will only be done once on first iteration
            if gsum.count == 0 && (method != .sgd || momentum > 0.0) {
                // only vanilla sgd doesnt need either lists
                // momentum needs gsum
                // adagrad needs gsum
                // adam and adadelta needs gsum and xsum
                for i in 0 ..< pglist.count {
                    
                    gsum.append(ArrayUtils.zerosFloat(pglist[i].params.count))
                    if method == SolverType.adam || method == SolverType.adadelta {
                        xsum.append(ArrayUtils.zerosFloat(pglist[i].params.count))
                    } else {
                        xsum.append([]) // conserve memory
                    }
                }
            }
            
            // perform an update for all sets of weights
            for i in 0 ..< pglist.count {
                
                let pg = pglist[i] // param, gradient, other options in future (custom learning rate etc)
                var p = pg.params
                var g = pg.grads
                
                // learning rate for some parameters.
                let l2DecayMul = pg.l2DecayMul ?? 1.0
                let l1DecayMul = pg.l1DecayMul ?? 1.0
                let l2Decay = self.l2Decay * l2DecayMul
                let l1Decay = self.l1Decay * l1DecayMul
                
                let plen = p.count
                for j in 0 ..< plen {
                    
                    l2DecayLoss += l2Decay*p[j]*p[j]/2 // accumulate weight decay loss
                    l1DecayLoss += l1Decay*abs(p[j])
                    let l1grad = l1Decay * (p[j] > 0 ? 1 : -1)
                    let l2grad = l2Decay * (p[j])
                    
                    let gij = (l2grad + l1grad + g[j]) / Float(batchSize) // raw batch gradient
                    
                    var gsumi: [Float] = []
                    var xsumi: [Float] = []
                    
                    if method != .sgd || momentum > 0.0 {
                        gsumi = gsum[i]
                        xsumi = xsum[i]
                    }
                    
                    switch method {
                    case .adam:
                        // adam update
                        gsumi[j] = gsumi[j] * β1 + (1-β1) * gij // update biased first moment estimate
                        xsumi[j] = xsumi[j] * β2 + (1-β2) * gij * gij // update biased second moment estimate
                        let biasCorr1 = gsumi[j] / (1 - pow(β1, Float(k))) // correct bias first moment estimate
                        let biasCorr2 = xsumi[j] / (1 - pow(β2, Float(k))) // correct bias second moment estimate
                        let dx =  -learningRate * biasCorr1 / (sqrt(biasCorr2) + ε)
                        p[j] += dx
                    case .adagrad:
                        // adagrad update
                        gsumi[j] = gsumi[j] + gij * gij
                        let dx = -learningRate / sqrt(gsumi[j] + ε) * gij
                        p[j] += dx
                    case .windowgrad:
                        // this is adagrad but with a moving window weighted average
                        // so the gradient is not accumulated over the entire history of the run.
                        // it's also referred to as Idea #1 in Zeiler paper on Adadelta. Seems reasonable to me!
                        gsumi[j] = ρ * gsumi[j] + (1-ρ) * gij * gij
                        let dx = -learningRate / sqrt(gsumi[j] + ε) * gij // eps added for better conditioning
                        p[j] += dx
                    case .adadelta:
                        gsumi[j] = ρ * gsumi[j] + (1-ρ) * gij * gij
                        let dx = -sqrt((xsumi[j] + ε)/(gsumi[j] + ε)) * gij
                        xsumi[j] = ρ * xsumi[j] + (1-ρ) * dx * dx // yes, xsum lags behind gsum by 1.
                        p[j] += dx
                    case .nesterov:
                        var dx = gsumi[j]
                        gsumi[j] = gsumi[j] * momentum + learningRate * gij
                        dx = momentum * dx - (1.0 + momentum) * gsumi[j]
                        p[j] += dx
                    default:
                        // assume SGD
                        if momentum > 0.0 {
                            // momentum update
                            let dx = momentum * gsumi[j] - learningRate * gij // step
                            gsumi[j] = dx // back this up for next iteration of momentum
                            p[j] += dx // apply corrected gradient
                        } else {
                            // vanilla sgd
                            p[j] +=  -learningRate * gij
                        }
                    }
                    g[j] = Float(0.0) // zero out gradient so that we can begin accumulating anew
                }
                
                newParamsAndGradients.append(
                    ParamsAndGrads(params: &p, grads: &g, l1DecayMul: l1DecayMul, l2DecayMul: l2DecayMul)
                )
            }
            
            net.assignParamsAndGrads(newParamsAndGradients)
        }
        return (l1DecayLoss, l2DecayLoss)
    }
}

