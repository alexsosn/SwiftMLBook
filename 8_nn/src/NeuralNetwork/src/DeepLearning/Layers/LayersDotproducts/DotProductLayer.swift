
import Foundation

protocol DotProductLayer: HiddenLayer {
    var l1DecayMul: Float {get set}
    var l2DecayMul: Float {get set}
    var filters: [Volume] {get set}
    var biases: Volume {get set}
}

extension DotProductLayer {
    public var paramsAndGrads: [ParamsAndGrads] {
        get {
            var result: [ParamsAndGrads] = (0 ..< outDepth).map{i in
                ParamsAndGrads(params: &filters[i].values,
                               grads: &filters[i].gradients,
                               l1DecayMul: l1DecayMul,
                               l2DecayMul: l2DecayMul)
            }
            
            result.append(ParamsAndGrads(params: &biases.values,
                                         grads: &biases.gradients,
                                         l1DecayMul: 0.0,
                                         l2DecayMul: 0.0))
            return result
        }
        set {
            assert(filters.count + 1 == newValue.count)
            
            for i in 0 ..< outDepth {
                filters[i].values = newValue[i].params
                filters[i].gradients = newValue[i].grads
            }
            biases.values = newValue.last!.params
            biases.gradients = newValue.last!.grads
        }
    }
}
