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
                l1DecayMul: Float = Float(0.0),
                l2DecayMul: Float = Float(1.0),
                biasPref: Float = Float(0.0),
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

public class FullyConnectedLayer: HiddenLayer {
    public var outDepth: Int
    public var outSx: Int = 1
    public var outSy: Int = 1
    public var inAct: Volume?
    public var outAct: Volume?
    public var l1DecayMul: Float
    public var l2DecayMul: Float
    public var numInputs: Int
    public var filters: [Volume]
    public var biases: Volume
    
    
    public init(opt: FullyConnectedLayerOpt) {
        
        // required
        // ok fine we will allow 'filters' as the word as well
        
        outDepth = opt.numNeurons ?? opt.filters ?? 0
        
        // onal
        l1DecayMul = opt.l1DecayMul
        l2DecayMul = opt.l2DecayMul
        
        // computed
        numInputs = opt.numInputs()
        
        // initializations
        let bias = opt.biasPref
        filters = []
        for _ in 0 ..< outDepth {
            filters.append(Volume(sx: 1, sy: 1, depth: numInputs)) // Volumeumes should be different!
        }
        biases = Volume(sx: 1, sy: 1, depth: outDepth, c: bias)
        
        assert(outDepth != 0, "numNeurons or filters should be not nil")
    }
    
    public func forward(_ V: inout Volume, isTraining: Bool) -> Volume {
        inAct = V
        let A = Volume(sx: 1, sy: 1, depth: outDepth, c: 0.0)
        var Vw = V.w
        for i in 0 ..< outDepth {
            
            var a = Float(0.0)
            var wi = filters[i].w
            for d in 0 ..< numInputs {
                a += Vw[d] * wi[d] // for efficiency use Volumes directly for now
            }
            a += biases.w[i]
            A.w[i] = a
        }
        outAct = A
        return outAct!
    }
    
    public func backward() -> () {
        guard let V = inAct,
            let outAct = outAct else {
                return
        }
        V.dw = ArrayUtils.zerosFloat(V.w.count) // zero out the gradient in input Volume
        
        // compute gradient wrt weights and data
        for i in 0 ..< outDepth {
            
            let tfi = filters[i]
            let chainGrad = outAct.dw[i]
            for d in 0 ..< numInputs {
                
                V.dw[d] += tfi.w[d]*chainGrad // grad wrt input data
                tfi.dw[d] += V.w[d]*chainGrad // grad wrt params
            }
            biases.dw[i] += chainGrad
            filters[i] = tfi
        }
    }
    
    public func getParamsAndGrads() -> [ParamsAndGrads] {
        var response: [ParamsAndGrads] = []
        for i in 0 ..< outDepth {
            
            response.append(ParamsAndGrads(
                params: &filters[i].w,
                grads: &filters[i].dw,
                l1DecayMul: l1DecayMul,
                l2DecayMul: l2DecayMul))
        }
        response.append(ParamsAndGrads(
            params: &biases.w,
            grads: &biases.dw,
            l1DecayMul: 0.0,
            l2DecayMul: 0.0))
        return response
    }
    
    public func assignParamsAndGrads(_ paramsAndGrads: [ParamsAndGrads]) ->() {
        assert(filters.count + 1 == paramsAndGrads.count)
        
        for i in 0 ..< outDepth {
            filters[i].w = paramsAndGrads[i].params
            filters[i].dw = paramsAndGrads[i].grads
        }
        biases.w = paramsAndGrads.last!.params
        biases.dw = paramsAndGrads.last!.grads
    }
}

