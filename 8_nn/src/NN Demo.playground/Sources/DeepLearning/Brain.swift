import Foundation

// An agent is in state0 and does action0
// environment then assigns reward0 and provides new state, state1
// Experience nodes store all this information, which is used in the
// Q-learning update step

public class Experience {
    public var state0: Float
    public var action0: Int
    public var reward0: Float
    public var state1: Float
    
    public init(state0: Float,
                action0: Int,
                reward0: Float,
                state1: Float) {
        self.state0 = state0
        self.action0 = action0
        self.reward0 = reward0
        self.state1 = state1
    }
}

// A Brain object does all the magic.
// over time it receives some inputs and some rewards
// and its job is to set the outputs to maximize the expected reward
public struct BrainOpt {
    public var temporalWindow: Int
    public var experienceSize: Int
    public var startLearnThreshold: Int
    public var γ: Float
    public var learningStepsTotal: Int
    public var learningStepsBurnin: Int
    public var ε_min: Float
    public var ε_test_time: Float
    public var randomActionDistribution: [Float]
    public var layerDefs: [LayerOptTypeProtocol]?
    public var hiddenLayerSizes: [Int]
    public var tdsolverOptions: SolverOpt?
    
    public init(temporalWindow: Int = 1,
                experienceSize: Int = 30000,
                startLearnThreshold: Int = 1000,
                γ: Float = 0.8,
                learningStepsTotal: Int = 100000,
                learningStepsBurnin: Int = 3000,
                ε_min: Float = Float(0.05),
                ε_test_time: Float = Float(0.01),
                randomActionDistribution: [Float] = [],
                layerDefs: [LayerOptTypeProtocol]? = nil,
                hiddenLayerSizes: [Int] = [],
                tdsolverOptions: SolverOpt? = nil){
        
        self.temporalWindow = temporalWindow
        self.experienceSize = experienceSize
        self.startLearnThreshold = startLearnThreshold
        self.γ = γ
        self.learningStepsTotal = learningStepsTotal
        self.learningStepsBurnin = learningStepsBurnin
        self.ε_min = ε_min
        self.ε_test_time = ε_test_time
        self.randomActionDistribution = randomActionDistribution
        self.layerDefs = layerDefs
        self.hiddenLayerSizes = hiddenLayerSizes
        self.tdsolverOptions = tdsolverOptions
    }
}

public class Brain {
    public var temporalWindow: Int
    public var experienceSize: Int
    public var startLearnThreshold: Int
    public var γ: Float
    public var learningStepsTotal: Int
    public var learningStepsBurnin: Int
    public var ε_min: Float
    public var ε_test_time: Float
    public var numActions: Int
    public var randomActionDistribution: [Float]
    public var netInputs: Int
    public var numStates: Int
    public var windowSize: Int
    public var stateWindow: [Float]
    public var actionWindow: [Int]
    public var rewardWindow: [Float]
    public var netWindow: [Float]
    public var valueNet: Net
    public var tdsolver: Solver
    public var experience: [Experience]
    public var age: Int
    public var forwardPasses: Int
    public var ε: Float
    public var latestReward: Float
    public var lastInputArray: [Float]
    public var averageRewardWindow: Window
    public var averageLossWindow: Window
    public var learning: Bool
    //    var policy: Int
    
    public init (numStates: Int, numActions: Int, opt: BrainOpt?) {
        let opt = opt ?? BrainOpt()
        // in number of time steps, of temporal memory
        // the ACTUAL input to the net will be (x,a) temporalWindow times, and followed by current x
        // so to have no information from previous time step going into value function, set to 0.
        temporalWindow = opt.temporalWindow
        // size of experience replay memory
        experienceSize = opt.experienceSize
        // number of examples in experience replay memory before we begin learning
        startLearnThreshold = opt.startLearnThreshold
        // gamma is a crucial parameter that controls how much plan-ahead the agent does. In [0,1]
        γ = opt.γ
        
        // number of steps we will learn for
        learningStepsTotal = opt.learningStepsTotal
        // how many steps of the above to perform only random actions (in the beginning)?
        learningStepsBurnin = opt.learningStepsBurnin
        // what ε value do we bottom out on? 0.0 => purely deterministic policy at end
        ε_min = opt.ε_min
        // what ε to use at test time? (i.e. when learning is disabled)
        ε_test_time = opt.ε_test_time
        
        // advanced feature. Sometimes a random action should be biased towards some values
        // for example in flappy bird, we may want to choose to not flap more often
        // this better sum to 1 by the way, and be of length numActions
        randomActionDistribution = opt.randomActionDistribution
        precondition( opt.randomActionDistribution.count == numActions,
                      "TROUBLE. randomActionDistribution should be same length as numActions.")
        
        var a = randomActionDistribution
        var s = Float(0.0)
        for k: Int in 0 ..< a.count {
            s += a[k]
        }
        assert( abs(s-1.0)<=0.0001,
                "TROUBLE. randomActionDistribution should sum to 1!")
        
        // states that go into neural net to predict optimal action look as
        // x0,a0,x1,a1,x2,a2,...xt
        // this variable controls the size of that temporal window. Actions are
        // encoded as 1-of-k hot vectors
        let netInputs = numStates * temporalWindow + numActions * temporalWindow + numStates
        self.netInputs = netInputs
        self.numStates = numStates
        self.numActions = numActions
        windowSize = max(temporalWindow, 2) // must be at least 2, but if we want more context even more
        stateWindow = ArrayUtils.zerosFloat(windowSize)
        actionWindow = ArrayUtils.zerosInt(windowSize)
        rewardWindow = ArrayUtils.zerosFloat(windowSize)
        netWindow = ArrayUtils.zerosFloat(windowSize)
        
        // create [state -> value of all possible actions] modeling net for the value function
        var layerDefs: [LayerOptTypeProtocol] = []
        if opt.layerDefs != nil {
            // this is an advanced usage feature, because size of the input to the network, and number of
            // actions must check out. This is not very pretty Object Oriented programming but I can't see
            // a way out of it :(
            layerDefs = opt.layerDefs!
            
            assert(layerDefs.count >= 2, "TROUBLE! must have at least 2 layers")
            
            assert(layerDefs.first is InputLayerOpt,
                   "TROUBLE! first layer must be input layer!")
            
            assert(layerDefs.last is RegressionLayerOpt,
                   "TROUBLE! last layer must be input regression!")
            
            let first = layerDefs.first as! LayerOutOptProtocol
            
            assert(first.outDepth * first.outSx * first.outSy == netInputs,
                   "TROUBLE! Number of inputs must be numStates * temporalWindow + numActions * temporalWindow + numStates!")
            
            let last = layerDefs.last as! RegressionLayerOpt
            
            assert(last.numInputs() == numActions,
                   "TROUBLE! Number of regression neurons should be numActions!")
        } else {
            // create a very simple neural net by default
            layerDefs.append(InputLayerOpt(outSx: 1, outSy: 1, outDepth: netInputs))
            // allow user to specify this via the option, for convenience
            var hl = opt.hiddenLayerSizes
            for k: Int in 0 ..< hl.count {
                layerDefs.append(FullyConnectedLayerOpt(numNeurons: hl[k], activation: .relu)) // relu by default
            }
            
            // WARNING: was: RegressionLayerOpt(numNeurons: numActions)
            layerDefs.append(RegressionLayerOpt()) // value function output
        }
        valueNet = Net(layerDefs)
        
        // and finally we need a Temporal Difference Learning solver!
        var tdsolverOptions = SolverOpt()
        tdsolverOptions.learningRate = Float(0.01)
        tdsolverOptions.momentum = Float(0.0)
        tdsolverOptions.batchSize = 64
        tdsolverOptions.l2Decay = Float(0.01)
        if opt.tdsolverOptions != nil {
            tdsolverOptions = opt.tdsolverOptions! // allow user to overwrite this
        }
        tdsolver = Solver(net: valueNet, options: tdsolverOptions)
        
        // experience replay
        experience = []
        
        // various housekeeping variables
        age = 0 // incremented every backward()
        forwardPasses = 0 // incremented every forward()
        ε = Float(1.0) // controls exploration exploitation tradeoff. Should be annealed over time
        latestReward = 0
        lastInputArray = []
        averageRewardWindow = Window(size: 1000, minsize: 10)
        averageLossWindow = Window(size: 1000, minsize: 10)
        learning = true
    }
    
    public func randomAction() -> Int? {
        // a bit of a helper function. It returns a random action
        // we are abstracting this away because in future we may want to
        // do more sophisticated things. For example some actions could be more
        // or less likely at "rest"/default state.
        if randomActionDistribution.count == 0 {
            return Random.Uniform.int(0, numActions)
        } else {
            // okay, lets do some fancier sampling:
            let p = Random.Uniform.float(0, 1.0)
            var cumprob = Float(0.0)
            for k: Int in 0 ..< numActions {
                cumprob += randomActionDistribution[k]
                if p < cumprob {
                    return k
                }
            }
        }
        return nil
    }
    
    public struct Policy {
        public var action: Int
        public var value: Float
    }
    
    public func policy(_ s: [Float]) -> Policy {
        // compute the value of doing any action in this state
        // and return the argmax action and its value
        var svol = Volume(sx: 1, sy: 1, depth: netInputs)
        svol.w = s
        let actionValues = valueNet.forward(&svol)
        var maxk = 0
        var maxval = actionValues.w[0]
        for k: Int in 1 ..< numActions {
            if actionValues.w[k] > maxval {
                maxk = k
                maxval = actionValues.w[k] }
        }
        return Policy(action: maxk, value: maxval)
    }
    
    public func getNetInput(_ xt: [Float]) -> [Float] {
        // return s = (x,a,x,a,x,a,xt) state vector.
        // It's a concatenation of last windowSize (x,a) pairs and current state x
        var w: [Float] = []
        w.append(contentsOf: xt) // start with current state
        // and now go backwards and append states and actions from history temporalWindow times
        let n = windowSize
        for k: Int in 0 ..< temporalWindow {
            // state
            w.append(stateWindow[n-1-k])
            // action, encoded as 1-of-k indicator vector. We scale it up a bit because
            // we dont want weight regularization to undervalue this information, as it only exists once
            var action1ofk = [Float](repeating: 0, count: numActions)
            action1ofk[actionWindow[n-1-k]] = Float(numStates)
            w.append(contentsOf: action1ofk)
        }
        return w
    }
    
    public func forward(_ inputArray: [Float]) -> Int {
        // compute forward (behavior) pass given the input neuron signals from body
        forwardPasses += 1
        lastInputArray = inputArray // back this up
        
        // create network input
        var action: Int
        var netInput: [Float]
        if forwardPasses > temporalWindow {
            // we have enough to actually do something reasonable
            netInput = getNetInput(inputArray)
            if learning {
                // compute ε for the ε-greedy policy
                ε = min(1.0, max(ε_min, 1.0-(Float(age) - Float(learningStepsBurnin))/(Float(learningStepsTotal) - Float(learningStepsBurnin))))
            } else {
                ε = ε_test_time // use test-time value
            }
            let rf = Random.Uniform.float(0,1)
            if rf < ε {
                // choose a random action with ε probability
                action = randomAction()!
            } else {
                // otherwise use our policy to make decision
                let maxact = policy(netInput)
                action = maxact.action
            }
        } else {
            // pathological case that happens first few iterations
            // before we accumulate windowSize inputs
            netInput = []
            action = randomAction()!
        }
        
        // remember the state and action we took for backward pass
        netWindow.removeFirst()
        netWindow.append(contentsOf: netInput)
        stateWindow.removeFirst()
        stateWindow.append(contentsOf: inputArray)
        actionWindow.removeFirst()
        actionWindow.append(action)
        
        return action
    }
    
    public func backward(_ reward: Float) {
        latestReward = reward
        averageRewardWindow.add(reward)
        rewardWindow.removeFirst()
        rewardWindow.append(reward)
        
        if !learning { return }
        
        // various book-keeping
        age += 1
        
        // it is time t+1 and we have to store (s_t, a_t, r_t, s_{t+1}) as new experience
        // (given that an appropriate number of state measurements already exist, of course)
        if forwardPasses > temporalWindow + 1 {
            let n = windowSize
            let e = Experience(
                state0: netWindow[n-2],
                action0: actionWindow[n-2],
                reward0: rewardWindow[n-2],
                state1: netWindow[n-1])
            if experience.count < experienceSize {
                experience.append(e)
            } else {
                // replace. finite memory!
                let ri = Random.Uniform.int(0, experienceSize)
                experience[ri] = e
            }
        }
        
        // learn based on experience, once we have some samples to go on
        // this is where the magic happens...
        if experience.count > startLearnThreshold {
            var avcost = Float(0.0)
            for _: Int in 0 ..< tdsolver.batchSize {
                let re = Random.Uniform.int(0, experience.count)
                let e = experience[re]
                var x = Volume(sx: 1, sy: 1, depth: netInputs)
                x.w = [e.state0]
                let maxact = policy([e.state1])
                let r = e.reward0 + γ * maxact.value
                let ystruct = (dim: e.action0, val: r)
                let loss = tdsolver.train(x: &x, y: ystruct)
                avcost += loss.loss
            }
            avcost = Float(avcost)/Float(tdsolver.batchSize)
            averageLossWindow.add(avcost)
        }
    }
    
    public func visSelf() -> String {
        // elt is a DOM element that this function fills with brain-related information
        
        // basic information
        let t = "experience replay size: \(experience.count) <br>" +
            "exploration epsilon: \(ε)<br>" +
            "age: \(age)<br>" +
            "average Q-learning loss: \(averageLossWindow.average())<br />" +
        "smooth-ish reward: \(averageRewardWindow.average())<br />"
        let brainvis = "<div><div>\(t)</div></div>"
        
        return brainvis
    }
}

// ----------------- Utilities -----------------
// contains various utility functions

// a window stores _size_ number of values
// and returns averages. Useful for keeping running
// track of validation or training accuracy during SGD
public class Window {
    public var v: [Float] = []
    public var size = 100
    public var minsize = 20
    public var sum = Float(0.0)
    
    public init(size: Int, minsize: Int) {
        v = []
        self.size = size
        self.minsize = minsize
        sum = 0
    }
    
    public func add(_ x: Float) {
        v.append(x)
        sum += x
        if v.count>size {
            let xold = v.removeFirst()
            sum -= xold
        }
    }
    
    public func average() -> Float {
        if v.count < minsize {
            return -1
        } else  {
            return Float(sum)/Float(v.count)
        }
    }
    
    public func reset() {
        v = []
        sum = 0
    }
}



