
import Foundation

enum MotionType: Int {
    case run
    case walk
    case shakePhone
    case squat
    case stand
    case __totalCount
    
    static func numberOfCases() -> Int {
        return MotionType.__totalCount.rawValue
    }
    
    func name() -> String {
        switch self {
        case .run:
            return "Run"
        case .walk:
            return "Walk"
        case .shakePhone:
            return "Shake Phone"
        case .squat:
            return "Squat"
        case .stand:
            return "Stand"
        case .__totalCount:
            fatalError()
        }
    }
}

enum SensorType: String {
    case Accelerometer
    case Gyroscope
    case Magnetometer
    case DeviceMotion
}
