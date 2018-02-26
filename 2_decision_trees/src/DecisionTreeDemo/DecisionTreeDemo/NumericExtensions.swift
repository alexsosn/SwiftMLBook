
import Foundation

// Utility code
extension Int {
    init(_ value: Bool) {
        if value {
            self = 1
        } else {
            self = 0
        }
    }
}

extension Double {
    init(_ value: Bool) {
        if value {
            self = 1
        } else {
            self = 0
        }
    }
}
