//
//  ViewController.swift
//  Empathy
//
//  Created by Alex Sosnovshchenko on 7/6/17.
//  Copyright © 2017 Stigma Inc. All rights reserved.
//

import UIKit

class ViewController: UIViewController {
    
    @IBOutlet weak var imageView: UIImageView!
    @IBOutlet weak var networkInputView: UIImageView!
    @IBOutlet weak var result: UILabel!

    override func viewDidLoad() {
        super.viewDidLoad()
        title = "Choose a photo"
    }
    
    func takePhoto(source: UIImagePickerControllerSourceType) {
        if UIImagePickerController.isSourceTypeAvailable(.photoLibrary) {
            let imagePicker = UIImagePickerController()
            imagePicker.delegate = self
            imagePicker.sourceType = source
            imagePicker.allowsEditing = true
            self.present(imagePicker, animated: true, completion: nil)
        }
    }
    
    func classify(image: UIImage) {
        Classifier.shared.classifyFace(image: image.cgImage!) { (distribution, error) in
            DispatchQueue.main.async {
                
                let resultString = distribution
                    .map{ (arg) -> String in
                        let (expression, confidence) = arg
                        return expression.rawValue.capitalized + String(format: " : %.02f", confidence)
                    }
                    .reduce("") { $0 + "\n" + $1 }
                
                let resultText = distribution.first!.0.rawValue.capitalized
                let alert = UIAlertController(title: resultText, message: nil, preferredStyle: .alert)
                alert.addAction(UIAlertAction(title: "OK", style: .cancel, handler: nil))
                self.present(alert, animated: true, completion: nil)
                self.result.text = resultString
            }
        }
    }
    
    @IBAction func photoAction(_ sender: Any) {
        takePhoto(source: .camera)
    }
    
    @IBAction func libraryAction(_ sender: Any) {
        takePhoto(source: .photoLibrary)
    }
}

extension ViewController: UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [String : Any]) {
        guard let image = info[UIImagePickerControllerEditedImage] as? UIImage else {
            print("Not an image.")
            return
        }
        self.dismiss(animated: true, completion: nil)
        imageView.image = image
        guard let processedImage = image
            .scaleImage(toSize: CGSize(width: 64, height: 64))?
//            .addFrameForNetwork()?
            .grayscale() else {
            print("Bad image.")
            return
        }
        networkInputView.image = processedImage
        classify(image: processedImage)
    }
}
