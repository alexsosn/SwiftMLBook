//: Playground - noun: a place where people can play
#if os(macOS)
    import Cocoa
#elseif os(iOS)
    import UIKit
#endif

var (trainData, trainLabels, testData, testLabels, imageSize, imageTrainCount, imageTestCount) = MNIST.loadData()

print("Image size: ", imageSize)
print("Image count: ", imageTrainCount, imageTestCount)

let input = InputLayerOpt(outSx: imageSize, outSy: imageSize, outDepth: 1)// declare size of input
// output Volume is of size 32x32x3 here
let conv1 = ConvolutionLayerOpt(sx: 5, filters: 16, stride: 1, pad: 2, activation: .relu)
// the layer will perform convolution with 16 kernels, each of size 5x5.
// the input will be padded with 2 pixels on all sides to make the output Volume of the same size
// output Volume will thus be 32x32x16 at this point
let pool1 = PoolingLayerOpt(sx: 2, stride: 2)
// output Volume is of size 16x16x16 here
let conv2 = ConvolutionLayerOpt(sx: 5, filters: 20, stride: 1, pad: 2, activation: .relu)
// output Volume is of size 16x16x20 here
let pool2 = PoolingLayerOpt(sx: 2, stride: 2)
// output Volume is of size 8x8x20 here
let conv3 = ConvolutionLayerOpt(sx: 5, filters: 20, stride: 1, pad: 2, activation: .relu)
// output Volume is of size 8x8x20 here
let pool3 = PoolingLayerOpt(sx: 2, stride: 2)
// output Volume is of size 4x4x20 here
let fc = FullyConnectedLayerOpt(numNeurons: 320, activation: .relu)

let softmax = SoftmaxLayerOpt(numClasses: 10)
// output Volume is of size 1x1x10 here

let net = Net([input, conv1, pool1, conv2, pool2, conv3, pool3, fc, softmax])

var solverOpts = SolverOpt()
solverOpts.learningRate = 1
solverOpts.method = .adadelta
solverOpts.batchSize = 50
solverOpts.l2Decay = Float(0.001)
solverOpts.l1Decay = Float(0.001)

let solver = Solver(net: net, options: solverOpts)

for (i, (x, y)) in zip(trainData, trainLabels).enumerated() {
    var volume = Volume(array: x)
    let res = solver.train(x: &volume, y: y)
    res.softmaxLoss
    //    print(i, res)
    
    if i % 100 == 0 {
        var test = Volume(array: testData[0])
        let loss = net.getCostLoss(V: &test, y: testLabels[0])
        print("Test loss", loss)
    }
}

//#if os(macOS)
//    import PlaygroundSupport
//
//    extension Volume: PlaygroundLiveViewable {
//        public var playgroundLiveViewRepresentation: PlaygroundLiveViewRepresentation {
//            return self.toImage()!
//        }
//    }
//
//    extension Volume {
//
//        public func toImage() -> NSImage? {
//
////            let intDenormArray: [UInt8] = w.map { (elem: Float) -> UInt8 in
////                return denormalize(elem)
////            }
//
//            let width = sx
//            let height = sy
//            let components = depth
//            let bitsPerComponent: Int = 8
//            let bitsPerPixel = bitsPerComponent * components
//            let bytesPerRow = (components * width)
//            let bitmapInfo: CGBitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue)
//            let colorSpace = CGColorSpaceCreateDeviceRGB()
//            let providerRef = CGDataProvider(
//                data: Data(bytes: UnsafePointer<UInt8>(intDenormArray), count: intDenormArray.count * components) as CFData
//            )
//
//            guard let cgim = CGImage(
//                width: width,
//                height: height,
//                bitsPerComponent: bitsPerComponent,
//                bitsPerPixel: bitsPerPixel,
//                bytesPerRow: bytesPerRow,
//                space: colorSpace,
//                bitmapInfo: bitmapInfo,
//                provider: providerRef!,
//                decode: nil,
//                shouldInterpolate: false,
//                intent: .defaultIntent
//                )  else {
//                    return nil
//            }
//
//            let newImage = UIImage(cgImage: cgim)
//            return newImage
//        }
//    }
//#endif


