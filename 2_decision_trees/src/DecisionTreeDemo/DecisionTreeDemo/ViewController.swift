
import UIKit

class ViewController: UIViewController {
    override func viewDidLoad() {
        super.viewDidLoad()
        let flow = Flow()
        // Parse CSV file and make predictions using pretrained Scikit-learn decision tree and random forest.
        // The output is printed to console.
        flow.runWithCSV()
        
        // 0. Generate fake dataset of 50 samples and split it into training and test set.
        // 1. Train GameplayKit decision tree with the training set and validate it on the test set.
        // As for iOS 11.2.5 training causes stack overflow error even with as little data in training set as 41*6 matrix.
        // 2. Make predictions on the same data using pretrained Scikit-learn decision tree and random forest.
        // The output is printed to console.
        flow.run(50)

    }
}
