//
//  Layer.swift
//  ConvNetSwift
//
import Foundation

public struct ParamsAndGrads {
    public var params: [Float]
    public var grads: [Float]
    public var l1DecayMul: Float?
    public var l2DecayMul: Float?
    
    public init(params: inout [Float],
                grads: inout [Float],
                l1DecayMul: Float,
                l2DecayMul: Float) {
        self.params = params
        self.grads = grads
        self.l1DecayMul = l1DecayMul
        self.l2DecayMul = l2DecayMul
    }
}

public protocol Layer {
    var outWidth: Int {get set}
    var outHeight: Int {get set}
    var outDepth: Int {get set}
    var outActivations: Volume? {get set}
    var paramsAndGrads: [ParamsAndGrads] {get set}
    func forward(_ vol: inout Volume, isTraining: Bool) -> Volume
}

extension Layer {
    public var paramsAndGrads: [ParamsAndGrads] {
        get {
            return []
        }
        set {}
    }
}

public protocol HiddenLayer: Layer {
    func backward()
}

// Layers that implement a loss. Currently these are the layers that
// can initiate a backward() pass. In future we probably want a more
// flexible heightstem that can accomodate multiple losses to do multi-task
// learning, and stuff like that. But for now, one of the layers in this
// file must be the final layer in a Net.

public protocol LossLayer: Layer {
    func backward(_ y: Int) -> Float
}

public enum ActivationType: String {
    case undefined
    case relu
    case sigmoid
    case tanh
    case maxout
}
