//
//  ViewController.swift
//  VectorBot
//
//  Created by Oleksandr on 6/21/17.
//  Copyright © 2017 OWL. All rights reserved.
//

import UIKit

class BotsTableViewController: UITableViewController {
    
    override func viewDidLoad() {
        super.viewDidLoad()
        self.title = "Choose a bot"
        // Do any additional setup after loading the view, typically from a nib.
        
    }
    
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        
        guard let identifier = segue.identifier else {
            return
        }
        
        guard let chatController = segue.destination as? ChatViewController else {
            return
        }
        
        var fileName = ""
        var avatar: UIImage
        var allLowercase = false
        
        switch identifier {
        case "shakespeare":
            fileName = "WilliamShakespeare"
            avatar = #imageLiteral(resourceName: "Shakespeare")
        case "franklin":
            fileName = "BenjaminFranklin"
            avatar = #imageLiteral(resourceName: "BenFranklin")
        case "twain":
            fileName = "MarkTwain"
            avatar = #imageLiteral(resourceName: "Mark_Twain")
        case "galsworthy":
            fileName = "JohnGalsworthy"
            avatar = #imageLiteral(resourceName: "John_Galsworthy")
        case "churchill":
            fileName = "WinstonChurchill"
            avatar = #imageLiteral(resourceName: "Sir_Winston_Churchill")
        case "news":
            fileName = "out"
            avatar = #imageLiteral(resourceName: "bot")
            allLowercase = true
        default:
            fatalError()
        }
        
        loadData(fileName: fileName) { (provider) in
            chatController.word2VecProvider = provider
            chatController.avatar = avatar
            chatController.allLowercase = allLowercase
        }
    }
    
    private func loadData(fileName: String, successCallback: @escaping (W2VDistance)->()) {
        print(NSDate(), "Started loading data...")
        DispatchQueue.global().async() {
            let url = Bundle.main.url(forResource: fileName, withExtension: "bin")!
            let word2VecProvider = W2VDistance()
            var error: NSError?
            word2VecProvider.loadBinaryVectorFile(url, error: &error)
            
            if let error = error {
                print("Error loading file: ", error)
            }
            
            DispatchQueue.main.async() { [word2VecProvider] in
                successCallback(word2VecProvider)
            }
        }
    }
}

