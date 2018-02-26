import Foundation

// - FullyConn is fully connected dot products

public struct FullyConnectedLayerOpt: LayerInOptProtocol, LayerOptActivationProtocol {
    public var numNeurons: Int?
    public var filters: Int?
    public var inSx: Int
    public var inSy: Int
    public var inDepth: Int
    public var l1DecayMul: Float
    public var l2DecayMul: Float
    public var biasPref: Float
    public var activation: ActivationType
    public var dropProb: Float?
    
    /// numNeurons or filters should be not nil
    public init(numNeurons: Int? = nil,
                filters: Int? = nil,
                inSx: Int = 0,
                inSy: Int = 0,
                inDepth: Int = 0,
                l1DecayMul: Float = 0.0,
                l2DecayMul: Float = 1.0,
                biasPref: Float = 0.0,
                activation: ActivationType = .undefined,
                dropProb: Float? = nil) {
        self.numNeurons = numNeurons
        self.filters = filters
        self.inSx = inSx
        self.inSy = inSy
        self.inDepth = inDepth
        self.l1DecayMul = l1DecayMul
        self.l2DecayMul = l2DecayMul
        self.biasPref = biasPref
        self.activation = activation
        self.dropProb = dropProb
    }
}

public class FullyConnectedLayer: DotProductLayer {
    public var outDepth: Int
    public var outWidth: Int = 1
    public var outHeight: Int = 1
    public var inActivations: Volume?
    public var outActivations: Volume?
    public var l1DecayMul: Float
    public var l2DecayMul: Float
    public var numInputs: Int
    public var filters: [Volume]
    public var biases: Volume
    
    
    public init(opt: FullyConnectedLayerOpt) {
        outDepth = opt.numNeurons ?? opt.filters ?? 0
        l1DecayMul = opt.l1DecayMul
        l2DecayMul = opt.l2DecayMul
        numInputs = opt.numInputs()
        
        let bias = opt.biasPref
        // Volumes should be different!
        filters = (0 ..< outDepth).map{_ in Volume(width: 1, height: 1, depth: opt.numInputs())}
        biases = Volume(width: 1, height: 1, depth: outDepth, fill: bias)
        
        assert(outDepth != 0, "numNeurons or filters should be not nil")
    }
    
    public func forward(_ volume: inout Volume, isTraining: Bool) -> Volume {
        inActivations = volume
        let outActivations = Volume(width: 1, height: 1, depth: outDepth, fill: 0.0)
        for i in 0 ..< outDepth {
            
            var a = Float(0.0)
            let filter = filters[i]
            for d in 0 ..< numInputs {
                a += volume.values[d] * filter.values[d]
            }
            a += biases.values[i]
            outActivations.values[i] = a
        }
        self.outActivations = outActivations
        return outActivations
    }
    
    public func backward() {
        guard let inActivations = inActivations,
            let outActivations = outActivations else {
                return
        }
        // zero out the gradient in input Volume
        inActivations.gradients = ArrayUtils.zerosFloat(inActivations.values.count)
        
        // compute gradient wrt weights and data
        for i in 0 ..< outDepth {
            
            let filter = filters[i]
            let chainGradient = outActivations.gradients[i]
            for d in 0 ..< numInputs {
                // grad wrt input data
                inActivations.gradients[d] += filter.values[d]*chainGradient
                // grad wrt params
                filter.gradients[d] += inActivations.values[d]*chainGradient
            }
            biases.gradients[i] += chainGradient
            filters[i] = filter
        }
    }
}

