import Foundation
import MapKit

extension CLLocationCoordinate2D {
    func array() -> [Double] {
        return [Double(latitude), Double(longitude)]
    }
}
