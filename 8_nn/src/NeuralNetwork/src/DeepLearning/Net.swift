
// Net manages a set of layers
// For now constraints: Simple linear order of layers, first layer input last layer a cost layer
import Foundation

public class Net {
    
    public var layers: [Layer] = []
    public var layerResponseLengths: [Int] = []
    
    // desugar layerDefs for adding activation, dropout layers etc
    public func desugar(_ defs: [LayerOptTypeProtocol]) -> [LayerOptTypeProtocol] {
        var newOptions: [LayerOptTypeProtocol] = []
        
        for i in 0 ..< defs.count {
            
            var def = defs[i]
            
            if let def = def as? ClassificationLayerOptProtocol {
                // add an fc layer here, there is no reason the user should
                // have to worry about this and we almost always want to
                let neuronNumber = def.numClasses
                newOptions.append(FullyConnectedLayerOpt(numNeurons: neuronNumber))
            } else if let def = def as? RegressionLayerOpt {
                // add an fc layer here, there is no reason the user should
                // have to worry about this and we almost always want to
                newOptions.append(FullyConnectedLayerOpt(numNeurons: def.numNeurons))
            } else if var mutableDef = def as? FullyConnectedLayerOpt {
                if mutableDef.activation == .relu {
                    mutableDef.biasPref = 0.1 // relus like a bit of positive bias to get gradients early
                    // otherwise it's technically possible that a relu unit will never turn on (by chance)
                    // and will never get any gradient and never contribute any computation. Dead relu.
                    def = mutableDef
                }
            } else if var mutableDef = def as? ConvolutionLayerOpt {
                if mutableDef.activation == .relu {
                    mutableDef.biasPref = 0.1 // relus like a bit of positive bias to get gradients early
                    // otherwise it's technically possible that a relu unit will never turn on (by chance)
                    // and will never get any gradient and never contribute any computation. Dead relu.
                    def = mutableDef
                }
            }
            
            newOptions.append(def)
            
            if def is LayerOptActivationProtocol {
                var def = def as! LayerOptActivationProtocol
                
                switch def.activation {
                case .undefined:
                    break
                case .relu:
                    newOptions.append(ReluLayerOpt())
                case .sigmoid:
                    newOptions.append(SigmoidLayerOpt())
                case .tanh:
                    newOptions.append(TanhLayerOpt())
                case .maxout:
                    // create maxout activation, and pass along group size, if provided
                    let def = def as! MaxoutLayerOpt
                    let gs = def.groupSize
                    newOptions.append(MaxoutLayerOpt(groupSize: gs))
                }
            }
            
            if let def = def as? FullyConnectedLayerOpt {
                if let prob = def.dropProb {
                    newOptions.append(DropoutLayerOpt(dropProb: prob))
                }
            }
        }
        return newOptions
    }
    
    // takes a list of layer definitions and creates the network layer objects
    public init(_ layerPrototypes: [LayerOptTypeProtocol]) {
        var defs = layerPrototypes
        // few checks
        assert(defs.count >= 2, "Error! At least one input layer and one loss layer are required.")
        assert(defs[0] is InputLayerOpt, "Error! First layer must be the input layer, to declare size of inputs")
        
        defs = desugar(defs)
        
        // create the layers
        layers = []
        for i in 0 ..< defs.count {
            
            var def = defs[i]
            
            if i>0 {
                var in_def = def as! LayerInOptProtocol
                var prev = layers[i-1]
                in_def.inSx = prev.outWidth
                in_def.inSy = prev.outHeight
                in_def.inDepth = prev.outDepth
                def = in_def
            }
            
            var layer: Layer?
            switch def {
            case is FullyConnectedLayerOpt:
                layer = FullyConnectedLayer(opt: def as! FullyConnectedLayerOpt)
            case is LocalResponseNormalizationLayerOpt:
                layer = LocalResponseNormalizationLayer(opt: def as! LocalResponseNormalizationLayerOpt)
            case is DropoutLayerOpt:
                layer = DropoutLayer(opt: def as! DropoutLayerOpt)
            case is InputLayerOpt:
                layer = InputLayer(opt: def as! InputLayerOpt)
            case is SoftmaxLayerOpt:
                layer = SoftmaxLayer(opt: def as! SoftmaxLayerOpt)
            case is RegressionLayerOpt:
                layer = RegressionLayer(opt: def as! RegressionLayerOpt)
            case is ConvolutionLayerOpt:
                layer = ConvolutionLayer(opt: def as! ConvolutionLayerOpt)
            case is PoolingLayerOpt:
                layer = PoolingLayer(opt: def as! PoolingLayerOpt)
            case is ReluLayerOpt:
                layer = ReluLayer(opt: def as! ReluLayerOpt)
            case is SigmoidLayerOpt:
                layer = SigmoidLayer(opt: def as! SigmoidLayerOpt)
            case is TanhLayerOpt:
                layer = TanhLayer(opt: def as! TanhLayerOpt)
            case is MaxoutLayerOpt:
                layer = MaxoutLayer(opt: def as! MaxoutLayerOpt)
            case is SVMLayerOpt:
                layer = SVMLayer(opt: def as! SVMLayerOpt)
            default:
                print("ERROR: UNRECOGNIZED LAYER TYPE: \(def)")
            }
            if layer != nil {
                layers.append(layer!)
            }
        }
    }
    
    // forward prop the network.
    // The solver class passes isTraining = true, but when this function is
    // called from outside (not from the solver), it defaults to prediction mode
    public func forward(_ volume: inout Volume, isTraining: Bool = false) -> Volume {
        
        var act = layers[0].forward(&volume, isTraining: isTraining)
        for i in 1 ..< layers.count {
            act = layers[i].forward(&act, isTraining: isTraining)
        }
        return act
    }
    
    public func getCostLoss(volume: inout Volume, y: Int) -> Float {
        _ = forward(&volume, isTraining: false)
        let loss = (layers.last! as! LossLayer).backward(y)
        return loss
    }
    
    public func getCostLoss(volume: inout Volume, y: Float) -> Float {
        _ = forward(&volume, isTraining: false)
        let loss = (layers.last! as! RegressionLayer).backward(y)
        return loss
    }
    
    // backprop: compute gradients wrt all parameters
    public func backward(_ y: Int) -> Float {
        let loss = (layers.last! as! LossLayer).backward(y)
        // last layer assumed to be loss layer
        let layerCount = layers.count
        for i in stride(from: layerCount-2, through: 0, by: -1) {
            // first layer assumed input
            (layers[i] as? HiddenLayer)?.backward()
        }
        return loss
    }
    
    public func backward(_ y: [Float]) -> Float {
        let loss = (layers.last! as! RegressionLayer).backward(y)
        // last layer assumed to be regression layer
        let N = layers.count
        for i in stride(from: N-2, through: 0, by: -1) {
            // first layer assumed input
            (layers[i] as? HiddenLayer)?.backward()
        }
        return loss
    }
    
    public func backward(_ y: Float) -> Float {
        let loss = (layers.last! as! RegressionLayer).backward(y)
        // last layer assumed to be regression layer
        let N = layers.count
        for i in stride(from: N-2, through: 0, by: -1) {
            // first layer assumed input
            (layers[i] as? HiddenLayer)?.backward()
        }
        return loss
    }
    
    public func backward(_ y: (dim: Int, val: Float)) -> Float {
        let loss = (layers.last! as! RegressionLayer).backward(y)
        // last layer assumed to be regression layer
        let N = layers.count
        for i in stride(from: N-2, through: 0, by: -1) {
            // first layer assumed input
            (layers[i] as? HiddenLayer)?.backward()
        }
        return loss
    }
    
    public func getParamsAndGrads() -> [ParamsAndGrads] {
        // accumulate parameters and gradients for the entire network
        var response: [ParamsAndGrads] = []
        layerResponseLengths = []
        
        for i in 0 ..< layers.count {
            
            var layer_reponse = layers[i].paramsAndGrads
            let layerRespLen = layer_reponse.count
            layerResponseLengths.append(layerRespLen)
            
            for j in 0 ..< layerRespLen {
                response.append(layer_reponse[j])
            }
        }
        return response
    }
    
    public func assignParamsAndGrads(_ paramsAndGrads: [ParamsAndGrads]) ->() {
        var offset = 0
        
        for i in 0 ..< layers.count {
            let length = layerResponseLengths[i]
            let chunk = Array(paramsAndGrads[offset ..< offset+length])
            layers[i].paramsAndGrads = chunk
            offset += length
        }
    }
    
    public func getPrediction() -> Int {
        // this is a convenience function for returning the argmax
        // prediction, assuming the last layer of the net is a softmax
        var softmax = layers[layers.count-1]
        assert(softmax is SoftmaxLayer, "getPrediction function assumes softmax as last layer of the net!")
        
        guard let outActivations = softmax.outActivations else {
            fatalError("softmax.outActivations is nil.")
        }
        
        var p = outActivations.values
        
        var maxv = p[0]
        var maxi = 0
        for i in 1 ..< p.count {
            
            if p[i] > maxv {
                maxv = p[i]
                maxi = i
            }
        }
        return maxi // return index of the class with highest class probability
    }
}

