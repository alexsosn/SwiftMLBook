
import Foundation

//: ### Declaring enums, which will serve as categorical data types.
enum Label: Int {
    case rabbosaurus
    case platyhog
}

protocol CategoricalFeature: RawRepresentable where RawValue == Int {
    static var allValuesCount: Int { get }
}

extension CategoricalFeature {
    func toOneHot() -> [Int] {
        return (0..<type(of: self).allValuesCount).map{ Int($0 == rawValue) }
    }
}

enum ColorFeature: Int, CategoricalFeature {
    case lightBlack
    case pinkGold
    case purplePolkaDot
    case spaceGray
    
    static let allValuesCount = ColorFeature.spaceGray.rawValue + 1
    
    init(_ string: String) {
        switch string {
        case "light black":
            self = .lightBlack
        case "pink gold":
            self = .pinkGold
        case "purple polka-dot":
            self = .purplePolkaDot
        case "space gray":
            self = .spaceGray
        default:
            fatalError()
        }
    }
}

struct FeatureLabels {
    //: The names of features.
    static let featureLabels: [String] = [ "Length", "Fluffy", "IsSpaceGray", "IsLightBlack", "IsPinkGold", "IsPurplePolkaDot"]
}
