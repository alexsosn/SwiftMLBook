
/*
 A MagicNet takes data: a list of convnetjs.Volume(), and labels
 which for now are assumed to be class indeces 0..K. MagicNet then:
 - creates data folds for cross-validation
 - samples candidate networks
 - evaluates candidate networks on all data folds
 - produces predictions by model-averaging the best networks
 */

import Foundation

public class MagicNet {
    
    public var data: [Volume]
    public var labels: [Int]
    public var trainRatio: Float
    public var numFolds: Int
    public var numCandidates: Int
    public var numEpochs: Int
    public var ensembleSize: Int
    public var batchSizeMin: Int
    public var batchSizeMax: Int
    public var l2DecayMin: Int
    public var l2DecayMax: Int
    public var learningRateMin: Int
    public var learningRateMax: Int
    public var momentumMin: Float
    public var momentumMax: Float
    public var neuronsMin: Int
    public var neuronsMax: Int
    public var folds: [Fold]
    public var candidates: [Candidate]
    public var evaluatedCandidates: [Candidate]
    public var uniqueLabels: [Int]
    public var iter: Int
    public var foldix: Int
    public var finishFoldCallback: (()->())?
    public var finishBatchCallback: (()->())?
    
    public struct Fold {
        public var train_ix: [Int]
        public var test_ix: [Int]
    }
    
    public struct Candidate {
        public var acc: [AnyObject]
        public var accv: Float
        public var layerDefs: [LayerOptTypeProtocol]
        public var solverDef: SolverOpt
        public var net: Net
        public var solver: Solver
    }
    
    public init(data:[Volume] = [], labels:[Int] = [], opt:[String: AnyObject]) {
        
        // required inputs
        self.data = data // store these pointers to data
        self.labels = labels
        
        // optional inputs
        trainRatio = opt["trainRatio"] as? Float ?? 0.7
        numFolds = opt["numFolds"] as? Int ?? 10
        numCandidates = opt["numCandidates"] as? Int ?? 50
        // we evaluate several in parallel
        // how many epochs of data to train every network? for every fold?
        // higher values mean higher accuracy in final results, but more expensive
        numEpochs = opt["numEpochs"] as? Int ?? 50
        // number of best models to average during prediction. Usually higher = better
        ensembleSize = opt["ensembleSize"] as? Int ?? 10
        
        // candidate parameters
        batchSizeMin = opt["batchSizeMin"] as? Int ?? 10
        batchSizeMax = opt["batchSizeMax"] as? Int ?? 300
        l2DecayMin = opt["l2DecayMin"] as? Int ?? -4
        l2DecayMax = opt["l2DecayMax"] as? Int ?? 2
        learningRateMin = opt["learningRateMin"] as? Int ?? -4
        learningRateMax = opt["learningRateMax"] as? Int ?? 0
        momentumMin = opt["momentumMin"] as? Float ?? 0.9
        momentumMax = opt["momentumMax"] as? Float ?? 0.9
        neuronsMin = opt["neuronsMin"] as? Int ?? 5
        neuronsMax = opt["neuronsMax"] as? Int ?? 30
        
        // computed
        folds = [] // data fold indices, gets filled by sampleFolds()
        candidates = [] // candidate networks that are being currently evaluated
        evaluatedCandidates = [] // history of all candidates that were fully evaluated on all folds
        uniqueLabels = ArrayUtils.arrUnique(labels)
        iter = 0 // iteration counter, goes from 0 -> numEpochs * numTrainingData
        foldix = 0 // index of active fold
        
        // callbacks
        finishFoldCallback = nil
        finishBatchCallback = nil
        
        // initializations
        if data.count > 0 {
            sampleFolds()
            sampleCandidates()
        }
    }
    
    // sets folds to a sampling of numFolds folds
    public func sampleFolds() -> () {
        let N = data.count
        let numTrain = Int(floor(trainRatio * Float(N)))
        folds = [] // flush folds, if any
        for _ in 0 ..< numFolds {
            var p = Random.permutation(N)
            let fold = Fold(
                train_ix: Array(p[0 ..< numTrain]),
                test_ix: Array(p[numTrain ..< N]))
            folds.append(fold)
        }
    }
    
    // returns a random candidate network
    public func sampleCandidate() -> Candidate {
        let inputDepth = data[0].w.count
        let numClasses = uniqueLabels.count
        
        // sample network topology and hyperparameters
        var layerDefs: [LayerOptTypeProtocol] = []
        let layerInput = InputLayerOpt(
            outSx: 1,
            outSy: 1,
            outDepth: inputDepth)
        layerDefs.append(layerInput)
        let nl = Int(Random.Weighted.sample([0,1,2,3], probabilities: [0.2, 0.3, 0.3, 0.2])!) // prefer nets with 1,2 hidden layers
        for _ in 0 ..< nl { // WARNING: iterator was q
            
            let ni = Random.Uniform.int(neuronsMin, neuronsMax)
            let actarr: [ActivationType] = [.tanh, .maxout, .relu]
            let act = actarr[Random.Uniform.int(0,3)]
            if Random.Uniform.float(0,1) < 0.5 {
                let dp = Random.Uniform.float()
                let layerFC = FullyConnectedLayerOpt(
                    numNeurons: ni,
                    activation: act,
                    dropProb: dp)
                layerDefs.append(layerFC)
            } else {
                let layerFC = FullyConnectedLayerOpt(
                    numNeurons: ni,
                    activation: act)
                layerDefs.append(layerFC
                )
            }
        }
        
        let layerSoftmax = SoftmaxLayerOpt(numClasses: numClasses)
        
        layerDefs.append(layerSoftmax)
        let net = Net(layerDefs)
        
        // sample training hyperparameters
        let bs = Random.Uniform.int(batchSizeMin, batchSizeMax) // batch size
        let l2 = pow(10, Random.Uniform.float(Float(l2DecayMin), Float(l2DecayMax))) // l2 weight decay
        let lr = pow(10, Random.Uniform.float(Float(learningRateMin), Float(learningRateMax))) // learning rate
        let mom = Random.Uniform.float(momentumMin, momentumMax) // momentum. Lets just use 0.9, works okay usually ;p
        let tp = Random.Uniform.float(0,1) // solver type
        var solverDef = SolverOpt()
        if tp < 0.33 {
            solverDef.method = .adadelta
            solverDef.batchSize = bs
            solverDef.l2Decay = l2
        } else if tp < 0.66 {
            solverDef.method = .adagrad
            solverDef.batchSize = bs
            solverDef.l2Decay = l2
            solverDef.learningRate = lr
        } else {
            solverDef.method = .sgd
            solverDef.batchSize = bs
            solverDef.l2Decay = l2
            solverDef.learningRate = lr
            solverDef.momentum = mom
        }
        
        let solver = Solver(net: net, options: solverDef)
        
        return Candidate(acc:[], accv: 0, layerDefs: layerDefs, solverDef: solverDef, net: net, solver: solver)
    }
    
    // sets candidates with numCandidates candidate nets
    public func sampleCandidates() -> () {
        candidates = [] // flush, if any
        for _ in 0 ..< numCandidates {
            
            let cand = sampleCandidate()
            candidates.append(cand)
        }
    }
    
    public func step() -> () {
        
        // run an example through current candidate
        iter += 1
        
        // step all candidates on a random data point
        let fold = folds[foldix] // active fold
        let dataix = fold.train_ix[Random.Uniform.int(0, fold.train_ix.count)]
        for k in 0 ..< candidates.count {
            
            var x = data[dataix]
            let l = labels[dataix]
            _ = candidates[k].solver.train(x: &x, y: l)
        }
        
        // process consequences: sample new folds, or candidates
        let lastiter = numEpochs * fold.train_ix.count
        if iter >= lastiter {
            // finished evaluation of this fold. Get final validation
            // accuracies, record them, and go on to next fold.
            var valAcc = evalValErrors()
            for k in 0 ..< candidates.count {
                
                var c = candidates[k]
                c.acc.append(valAcc[k] as AnyObject)
                c.accv += valAcc[k]
            }
            iter = 0 // reset step number
            foldix += 1 // increment fold
            
            if finishFoldCallback != nil {
                finishFoldCallback!()
            }
            
            if foldix >= folds.count {
                // we finished all folds as well! Record these candidates
                // and sample new ones to evaluate.
                for k in 0 ..< candidates.count {
                    
                    evaluatedCandidates.append(candidates[k])
                }
                // sort evaluated candidates according to accuracy achieved
                evaluatedCandidates.sort(by: { (a, b) -> Bool in
                    return (a.accv / Float(a.acc.count)) < (b.accv / Float(b.acc.count))
                }) // WARNING: not sure > or < ?
                
                // and clip only to the top few ones (lets place limit at 3*ensembleSize)
                // otherwise there are concerns with keeping these all in memory
                // if MagicNet is being evaluated for a very long time
                if evaluatedCandidates.count > 3 * ensembleSize {
                    let clip = Array(evaluatedCandidates[0 ..< 3*ensembleSize])
                    evaluatedCandidates = clip
                }
                if finishBatchCallback != nil {
                    finishBatchCallback!()
                }
                sampleCandidates() // begin with new candidates
                foldix = 0 // reset this
            } else {
                // we will go on to another fold. reset all candidates nets
                for k in 0 ..< candidates.count {
                    
                    var c = candidates[k]
                    let net = Net(c.layerDefs)
                    let solver = Solver(net: net, options: c.solverDef)
                    c.net = net
                    c.solver = solver
                }
            }
        }
    }
    
    public func evalValErrors() -> [Float] {
        // evaluate candidates on validation data and return performance of current networks
        // as simple list
        var vals: [Float] = []
        var fold = folds[foldix] // active fold
        for k in 0 ..< candidates.count {
            
            let net = candidates[k].net
            var v = Float(0.0)
            for q in 0 ..< fold.test_ix.count {
                
                var x = data[fold.test_ix[q]]
                let l = labels[fold.test_ix[q]]
                _ = net.forward(&x)
                let yhat = net.getPrediction()
                v += (yhat == l ? 1.0 : 0.0) // 0 1 loss
            }
            v /= Float(fold.test_ix.count) // normalize
            vals.append(v)
        }
        return vals
    }
    
    // returns prediction scores for given test data point, as Volume
    // uses an averaged prediction from the best ensembleSize models
    // x is a Volume.
    public func predictSoft(_ data: Volume) -> Volume {
        var data = data
        // forward prop the best networks
        // and accumulate probabilities at last layer into a an output Volume
        
        var evalCandidates: [Candidate] = []
        var nv = 0
        if evaluatedCandidates.count == 0 {
            // not sure what to do here, first batch of nets hasnt evaluated yet
            // lets just predict with current candidates.
            nv = candidates.count
            evalCandidates = candidates
        } else {
            // forward prop the best networks from evaluatedCandidates
            nv = min(ensembleSize, evaluatedCandidates.count)
            evalCandidates = evaluatedCandidates
        }
        
        // forward nets of all candidates and average the predictions
        var xout: Volume!
        var n: Int!
        for j in 0 ..< nv {
            
            let net = evalCandidates[j].net
            let x = net.forward(&data)
            if j==0 {
                xout = x
                n = x.w.count
            } else {
                // add it on
                for d in 0 ..< n {
                    
                    xout.w[d] += x.w[d]
                }
            }
        }
        // produce average
        for d in 0 ..< n {
            
            xout.w[d] /= Float(nv)
        }
        return xout
    }
    
    public func predict(_ data: Volume) -> Int {
        let xout = predictSoft(data)
        var predictedLabel: Int
        if xout.w.count != 0 {
            let stats = maxmin(xout.w)!
            predictedLabel = stats.maxi
        } else {
            predictedLabel = -1 // error out
        }
        return predictedLabel
        
    }
    
    // callback functions
    // called when a fold is finished, while evaluating a batch
    public func onFinishFold(_ f: (()->())?) -> () { finishFoldCallback = f; }
    // called when a batch of candidates has finished evaluating
    public func onFinishBatch(_ f: (()->())?) -> () { finishBatchCallback = f; }
    
}

