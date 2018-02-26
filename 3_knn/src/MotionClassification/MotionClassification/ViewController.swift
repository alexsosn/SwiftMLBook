
import UIKit

enum AppMode: Int {
    case train
    case predict
}

final class ViewController: UIViewController, UIPickerViewDelegate, UIPickerViewDataSource {
    
    @IBOutlet weak var segmentedControl: UISegmentedControl!
    @IBOutlet weak var picker: UIPickerView!
    @IBOutlet weak var recordButton: UIButton!
    
    var motionClassifier: MotionClassifier!

    override func viewDidLoad() {
        super.viewDidLoad()
        
        motionClassifier = MotionClassifier { [weak self] motionType in
            self?.pick(motionType: motionType)
        }
        
        recordButton.setTitle("Record", for: .normal)
        recordButton.setTitle("Pause", for: .selected)
    }
    
    @IBAction func segmentedControlValueChanged(_ sender: UISegmentedControl) {
        switch AppMode(rawValue: sender.selectedSegmentIndex) ?? .train {
        case .train:
            stopRecognition()
            recordButton.setTitle("Record", for: .normal)
            recordButton.setTitle("Pause", for: .selected)
            picker.isUserInteractionEnabled = true
            
        case .predict:
            stopDataRecording()
            recordButton.setTitle("Listen", for: .normal)
            recordButton.setTitle("Pause", for: .selected)
            picker.isUserInteractionEnabled = false
        }
    }
    
    @IBAction func recordButtonPressed(_ sender: Any) {
        recordButton.isSelected = !recordButton.isSelected

        switch AppMode(rawValue: segmentedControl.selectedSegmentIndex) ?? .train {
        case .train:
            recordButton.isSelected ? startDataRecording() : stopDataRecording()
        case .predict:
            recordButton.isSelected ? startRecognition() : stopRecognition()
        }
    }
    
    // MARK: - UIPickerViewDataSource
    func numberOfComponents(in pickerView: UIPickerView) -> Int {
        return 1
    }
    
    func pickerView(_ pickerView: UIPickerView, numberOfRowsInComponent component: Int) -> Int {
        return MotionType.numberOfCases()
    }
    
    func pickerView(_ pickerView: UIPickerView, titleForRow row: Int, forComponent component: Int) -> String? {
        return MotionType(rawValue:row)?.name()
    }
    
    func pickerView(_ pickerView: UIPickerView, didSelectRow row: Int, inComponent component: Int) {
        guard let selectedMotionType = MotionType(rawValue: picker.selectedRow(inComponent: 0)) else {
            fatalError()
        }
        motionClassifier.changeMotionType(newType: selectedMotionType)
    }
    
    // MARK: - Actions
    private func startDataRecording() {
        guard let selectedMotionType = MotionType(rawValue: picker.selectedRow(inComponent: 0)) else {
            fatalError()
        }
        motionClassifier.startTraining(motionType: selectedMotionType)
    }
    
    private func stopDataRecording() {
        motionClassifier.stopTraining()
    }
    
    private func startRecognition() {
        motionClassifier.startRecognition()
    }
    
    private func stopRecognition() {
        motionClassifier.stopRecognition()
    }
    
    private func pick(motionType: MotionType) {
        picker.selectRow(motionType.rawValue, inComponent: 0, animated: true)
    }
}

