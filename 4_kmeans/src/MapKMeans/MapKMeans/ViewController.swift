
import UIKit
import MapKit

class ViewController: UIViewController, MKMapViewDelegate, UIGestureRecognizerDelegate {
    
    @IBOutlet weak var mapView: MKMapView!
    
    var clusters = [Int]()
    var colors = [UIColor]()
    var centroidAnnotations = [MKPointAnnotation]()
    var savedAnnotations = [MKPointAnnotation]()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        colors = (0..<Settings.k).map{_ in Random.Uniform.randomColor()}
        mapView.delegate = self
        let gestureRecognizer = UITapGestureRecognizer(target: self, action:#selector(handleTapOnMap(gestureReconizer:)))
        gestureRecognizer.delegate = self
        mapView.addGestureRecognizer(gestureRecognizer)
    }
    
    @objc func handleTapOnMap(gestureReconizer: UILongPressGestureRecognizer) {
        let location = gestureReconizer.location(in: mapView)
        let coordinate = mapView.convert(location,toCoordinateFrom: mapView)
        
        // Adding annotation:
        let annotation = MKPointAnnotation()
        annotation.coordinate = coordinate
        mapView.addAnnotation(annotation)
        savedAnnotations.append(annotation)
    }
    
    func clusterize() {
        let k = Settings.k
        let randomInitialization = Settings.randomInitialization
        colors = (0..<k).map{_ in Random.Uniform.randomColor()}
        let data = savedAnnotations.map{ $0.coordinate.array() }
        var kMeans = KMeans(k: k)
        kMeans.initialization = randomInitialization ? .random : .plusplus
        clusters = kMeans.train(data: data)
        
        centroidAnnotations = kMeans.centroids
            .map { CLLocationCoordinate2D(latitude: $0[0], longitude: $0[1]) }
            .map { coordinate in
                let annotation = MKPointAnnotation()
                annotation.coordinate = coordinate
                annotation.title = "\(coordinate)"
                return annotation
        }
    }
    
    @IBAction func changeInitializationType(_ sender: UISwitch) {
        Settings.randomInitialization = !sender.isOn
    }
    
    @IBAction func showInfo(_ sender: Any) {
        let text = """
        Tap to add more data points.
        Toggle to switch between kMeans++ and random initialization.
        Press "Settings" to change k.
        Press ↻ to restart algorithm.
        Red pins are centroids of clusters.
        """
        
        let alertController = UIAlertController(title: "Info",
                                                message: text ,
                                                preferredStyle: .alert)
        
        let cancelAction = UIAlertAction(title: "Ok", style: .cancel, handler: nil)
        alertController.addAction(cancelAction)
        present(alertController, animated: true, completion: nil)
    }
    
    @IBAction func restart(_ sender: Any) {
        clusterize()
        mapView.removeAnnotations(mapView.annotations)
        mapView.addAnnotations(savedAnnotations)
        mapView.addAnnotations(centroidAnnotations)
    }
    
    @IBAction func openSettings(_ sender: Any) {
        let alertController = UIAlertController(title: "Select k", message: "Please, specify k for k-means algorithm:", preferredStyle: .alert)
        
        let confirmAction = UIAlertAction(title: "Confirm", style: .default) { [weak self] (_) in
            guard let `self` = self else { return }
            
            if let text = alertController.textFields?[0].text {
                let k = Int(text) ?? Settings.k
                Settings.k = k>0 ? k : Settings.k
                
                self.restart(self)
            }
        }
        
        let cancelAction = UIAlertAction(title: "Cancel", style: .cancel, handler: nil)
        
        alertController.addTextField { (textField) in
            textField.placeholder = "\(Settings.k)"
            textField.keyboardType = .numberPad
        }
        
        alertController.addAction(confirmAction)
        alertController.addAction(cancelAction)
        
        present(alertController, animated: true, completion: nil)
    }
    
    func mapView(_ mapView: MKMapView, viewFor annotation: MKAnnotation) -> MKAnnotationView? {
        let view = MKPinAnnotationView(annotation: annotation, reuseIdentifier: "pin")
        let annotations = savedAnnotations
        
        if clusters.count == annotations.count {
            if centroidAnnotations.contains(where: { (item) -> Bool in
                item.coordinate.latitude == annotation.coordinate.latitude
                    && item.coordinate.longitude == annotation.coordinate.longitude
            }) {
                view.pinTintColor = MKPinAnnotationView.redPinColor()
                view.animatesDrop = true
            } else {
                if let index = annotations.index(where: { (item) -> Bool in
                    item.coordinate.latitude == annotation.coordinate.latitude
                        && item.coordinate.longitude == annotation.coordinate.longitude
                }) {
                    view.pinTintColor = colors[clusters[index]]
                }
            }
        } else {
            view.pinTintColor = MKPinAnnotationView.purplePinColor()
        }
        return view
    }
}

