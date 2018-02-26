//
//  ImageUtils.swift
//  Empathy
//
//  Created by Alex Sosnovshchenko on 8/5/17.
//  Copyright © 2017 Stigma Inc. All rights reserved.
//

import Foundation
import UIKit
import CoreGraphics

extension UIImage {
    
    func scaleImage(toSize newSize: CGSize) -> UIImage? {
        let newRect = CGRect(x: 0, y: 0, width: newSize.width, height: newSize.height).integral
        UIGraphicsBeginImageContextWithOptions(newSize, false, 0)
        if let context = UIGraphicsGetCurrentContext() {
            context.interpolationQuality = .high
            let flipVertical = CGAffineTransform(a: 1, b: 0, c: 0, d: -1, tx: 0, ty: newSize.height)
            context.concatenate(flipVertical)
            context.draw(self.cgImage!, in: newRect)
            let newImage = UIImage(cgImage: context.makeImage()!)
            UIGraphicsEndImageContext()
            return newImage
        }
        return nil
    }
    
    func addFrameForNetwork() -> UIImage? {
        // Change true to false in order to replace black frame with transparent.
        UIGraphicsBeginImageContextWithOptions(CGSize(width: 64, height: 64), true, 0)
        let newRect = CGRect(x: 8, y: 8, width: 48, height: 48).integral
        self.draw(in: newRect, blendMode: .normal, alpha: 1.0)
        let resultImage =  UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return resultImage
    }
    
    // https://stackoverflow.com/a/41646252/2291058
    func grayscale() -> UIImage? {
        
        UIGraphicsBeginImageContextWithOptions(self.size, false, self.scale)
        let imageRect = CGRect(x: 0.0, y: 0.0, width: self.size.width, height: self.size.height)
        let context = UIGraphicsGetCurrentContext()
        
        // Draw a white background
        context!.setFillColor(red: 1.0, green: 1.0, blue: 1.0, alpha: 1.0)
        context!.fill(imageRect)
        
        // optional: increase contrast with colorDodge before applying luminosity
        // (my images were too dark when using just luminosity - you may not need this)
        self.draw(in: imageRect, blendMode: CGBlendMode.colorDodge, alpha: 0.7)
        
        
        // Draw the luminosity on top of the white background to get grayscale of original image
        self.draw(in: imageRect, blendMode: CGBlendMode.luminosity, alpha: 0.90)
        
        // optional: re-apply alpha if your image has transparency - based on user1978534's answer (I haven't tested this as I didn't have transparency - I just know this would be the the syntax)
        // self.draw(in: imageRect, blendMode: CGBlendMode.destinationIn, alpha: 1.0)
        
        let grayscaleImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return grayscaleImage
    }
}
