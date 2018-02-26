//: Play with convolutional filters

import UIKit
import Accelerate

func convolve(image: UIImage, filter: [Int16], divisor: Int) -> UIImage {
    precondition(filter.count == 9 || filter.count == 25 || filter.count == 49, "Kernel size must be 3x3, 5x5 or 7x7.")
    let kernelSide = UInt32(sqrt(Float(filter.count)))
    
    let imageRef = image.cgImage!
    
    let inBitmapData = imageRef.dataProvider!.data!
    let height = imageRef.height
    let width = imageRef.width
    let rowBytes = imageRef.bytesPerRow
    
    var inBuffer = vImage_Buffer(data: UnsafeMutablePointer(mutating: CFDataGetBytePtr(inBitmapData)), height: UInt(height), width: UInt(width), rowBytes: rowBytes)
    
    let pixelBuffer = malloc(rowBytes * height)
    
    var outBuffer = vImage_Buffer(data: pixelBuffer, height: UInt(height), width: UInt(width), rowBytes: rowBytes)
    
    var backgroundColor : Array<UInt8> = [0,0,0,0]
    
    _ = vImageConvolve_ARGB8888(&inBuffer, &outBuffer, nil, 0, 0, filter, kernelSide, kernelSide, Int32(divisor), &backgroundColor, UInt32(kvImageBackgroundColorFill))
    
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue).rawValue
    
    let context = CGContext(data: outBuffer.data, width: width, height: height, bitsPerComponent: 8, bytesPerRow: outBuffer.rowBytes, space: colorSpace, bitmapInfo: bitmapInfo)!
    
    let outCGimage = context.makeImage()!
    
    let outImage = UIImage(cgImage: outCGimage, scale: image.scale, orientation: .up)
    
    free(pixelBuffer)
    
    return outImage
}



let nüra = UIImage(named: "Nüra")!

let sharpenFilter: [Int16] = [0,-1,0,
                              -1,5,-1,
                              0,-1,0]

let sharpened = convolve(image: nüra, filter: sharpenFilter, divisor: 1)

let blurFilter: [Int16] = [1,4,7,4,1,
                           4,16,26,16,4,
                           7,26,31,26,7,
                           4,16,26,16,4,
                           1,4,7,4,1,]

let blurred = convolve(image: nüra, filter: blurFilter, divisor: 273)



let dino = UIImage(named: "Dino")!
let edgeDetectionFilter: [Int16] = [0,1,0,
                                    1,-4,1,
                                    0,1,0]

let edges = convolve(image: dino, filter: edgeDetectionFilter, divisor: 1)

let edgeDetectionFilter2: [Int16] = [-2,-2,-2,
                                    -2,16,-2,
                                    -2,-2,-2]

let edges2 = convolve(image: dino, filter: edgeDetectionFilter2, divisor: 1)

let embossFilter: [Int16] = [-2,-1,0,
                             -1,1,1,
                             0,1,2]

let embossed = convolve(image: dino, filter: embossFilter, divisor: 1)
