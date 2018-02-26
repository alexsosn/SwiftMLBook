//
//  AppDelegate.swift
//  VectorBot
//
//  Created by Oleksandr on 6/21/17.
//  Copyright © 2017 OWL. All rights reserved.
//

import UIKit
import Speech

@UIApplicationMain
class AppDelegate: UIResponder, UIApplicationDelegate {
    
    var window: UIWindow?
    
    func application(_ application: UIApplication, didFinishLaunchingWithOptions launchOptions: [UIApplicationLaunchOptionsKey: Any]?) -> Bool {
        SFSpeechRecognizer.requestAuthorization { authStatus in
            OperationQueue.main.addOperation {
                switch authStatus {
                case .authorized:
                    break
                case .denied:
                    fatalError("User denied access to speech recognition")
                case .restricted:
                    fatalError("Speech recognition restricted on this device")
                case .notDetermined:
                    fatalError("Speech recognition not yet authorized")
                }
            }
        }
        return true
    }
}

