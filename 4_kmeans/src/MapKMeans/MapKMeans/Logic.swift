
import Foundation

// This is needed for use with reduce on [Bool] arrays.
func and(_ lh: Bool, _ rh: Bool) -> Bool {
    return lh && rh
}

func or(_ lh: Bool, _ rh: Bool) -> Bool {
    return lh || rh
}

