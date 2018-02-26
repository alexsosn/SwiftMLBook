
let input = InputLayerOpt(outSx: 1, outSy: 1, outDepth: 2)
let softmax = SoftmaxLayerOpt(numClasses: 10)
let net = Net([input, softmax])

assert(net.layers.count == 3, "Additional layer should be added between input and softmax.")
assert(net.layers[1] is FullyConnectedLayer, "FC layer should be added between input and softmax.")
let fc0 = net.layers[1] as! FullyConnectedLayer

let filters = fc0.filters.flatMap{ $0.w }

print(filters)

for i in 0 ..< filters.count {
    for j in i+1 ..< filters.count {
        assert(filters[i] != filters[j], "All weights should be random and not equal to each other.")
    }
}
