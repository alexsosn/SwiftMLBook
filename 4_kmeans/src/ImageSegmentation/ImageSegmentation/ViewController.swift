
import UIKit

class ViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate {

    @IBOutlet private weak var activityIndicator: UIActivityIndicatorView!
    @IBOutlet private weak var imageView: UIImageView!
    private var image: UIImage?
    
    private var k = 5
    
    override func viewDidLoad() {
        super.viewDidLoad()
        activityIndicator.isHidden = true
        randomizeImage()
        presentAlert()
    }
    
    private func randomizeImage() {
        let imageCount = 6
        let images = (1...imageCount).map(String.init)
        let randIdx = Int(arc4random_uniform(UInt32(imageCount)))
        let image = UIImage(named: images[randIdx])
        self.image = image
        imageView.image = image
    }

    private func presentAlert() {
        let alertController = UIAlertController(title: "Select k", message: "Please, specify k for k-means algorithm:", preferredStyle: .alert)
        
        let confirmAction = UIAlertAction(title: "Confirm", style: .default) { [weak self] (_) in
            guard let `self` = self else { return }

            if let text = alertController.textFields?[0].text {
                let k = Int(text) ?? self.k
                self.k = k > 0 ? k : self.k

                self.image.flatMap(self.segment)
            }
        }
        
        let cancelAction = UIAlertAction(title: "Cancel", style: .cancel, handler: nil)
        
        alertController.addTextField { (textField) in
            textField.placeholder = "5"
            textField.keyboardType = .numberPad
        }
        
        alertController.addAction(confirmAction)
        alertController.addAction(cancelAction)
        
        present(alertController, animated: true, completion: nil)
    }
    
    @IBAction func chooseImage(_ sender: Any) {
        if UIImagePickerController.isSourceTypeAvailable(UIImagePickerControllerSourceType.photoLibrary) {
            let imagePicker = UIImagePickerController()
            imagePicker.delegate = self
            imagePicker.sourceType = UIImagePickerControllerSourceType.photoLibrary;
            imagePicker.allowsEditing = true
            present(imagePicker, animated: true, completion: nil)
        }
    }
    
    @IBAction func takePhote(_ sender: Any) {
        if UIImagePickerController.isSourceTypeAvailable(UIImagePickerControllerSourceType.camera) {
            let imagePicker = UIImagePickerController()
            imagePicker.delegate = self
            imagePicker.sourceType = UIImagePickerControllerSourceType.camera;
            imagePicker.allowsEditing = true
            present(imagePicker, animated: true, completion: nil)
        }
    }
    
    @IBAction func chooseRandomImage(_ sender: Any) {
        randomizeImage()
        if let image = self.image {
            segment(image: image)
        }
    }
    
    @IBAction func changeK(_ sender: Any) {
        presentAlert()
    }
    
    @IBAction func save(_ sender: Any) {
        guard let image = imageView.image else { return }
        guard let imageData = UIImageJPEGRepresentation(image, 0.6) else {
            print("Failed to get an image data.")
            return
        }
        guard let compressedJPGImage = UIImage(data: imageData) else {
            print("Failed to decode UIImage from compressed data.")
            return
        }
        UIImageWriteToSavedPhotosAlbum(compressedJPGImage, nil, nil, nil)
    }
    
    private func segment(image: UIImage) {
        self.image = image
        self.imageView.image = image
        
        activityIndicator.isHidden = false
        activityIndicator.startAnimating()
        
        DispatchQueue.global().async { [weak self, k = k] in
            guard let `self` = self else { return }

            let kMeans = KMeansQuantizer()
            kMeans.k = Int32(k)
            let segmented = kMeans.segment(image)
            
            DispatchQueue.main.async { [weak self] in
                guard let `self` = self else { return }
                self.imageView.image = segmented
                self.activityIndicator.stopAnimating()
                self.activityIndicator.isHidden = true
            }
        }
    }
    
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [String : Any]) {
        dismiss(animated: true, completion: nil);

        if let image = info[UIImagePickerControllerEditedImage] as? UIImage {
            segment(image: image)
        } else if let image = info[UIImagePickerControllerOriginalImage] as? UIImage {
            segment(image: image)
        } else {
            print("No image :(")
        }
    }
}

