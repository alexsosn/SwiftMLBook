
import Foundation
import CoreMotion

enum MotionInputError: Error {
    case noDeviceAvailable
}

final class MotionInput {
    private let manager = CMMotionManager()
    private var accelerationAccumulator: [[Double]] = []
    private let callback: ([[Double]])->()

    var seriesLength = 25
    var updateInterval = 0.1
    
    init(callback: @escaping (_ series: [[Double]])->()) {
        self.callback = callback
    }

    deinit {
        stop()
    }

    func start() throws {
        let allDevicesAvailable = manager.isAccelerometerAvailable
        
        guard allDevicesAvailable else { throw MotionInputError.noDeviceAvailable }
        
        manager.accelerometerUpdateInterval = updateInterval

        manager.startAccelerometerUpdates(to: OperationQueue.main) { [weak self] (data: CMAccelerometerData?, error: Error?) in
            guard let `self` = self else { return }
            let accumulatorCount = self.accelerationAccumulator.count
            precondition(accumulatorCount <= self.seriesLength)
            
            if let acceleration = data?.acceleration {
                let acceleration3Vec = [acceleration.x, acceleration.y, acceleration.z]
                self.accelerationAccumulator.append(acceleration3Vec)
                
                if accumulatorCount == self.seriesLength {
                    let tempAccumulator = self.accelerationAccumulator
                    print("Sample recorded.")
                    self.accelerationAccumulator = []
                    self.callback(tempAccumulator)
                }
            }
        }
    }
    
    func stop() {
        manager.stopAccelerometerUpdates()
    }
}
