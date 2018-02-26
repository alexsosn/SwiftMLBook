//
//  ChatTableViewController.swift
//  VectorBot
//
//  Created by Oleksandr on 6/22/17.
//  Copyright © 2017 OWL. All rights reserved.
//

import Foundation
import UIKit

class ChatViewController: UIViewController, UITableViewDelegate, UITableViewDataSource {
    @IBOutlet weak var recordButton: UIButton!
    @IBOutlet weak var tableView: UITableView!
    var avatar: UIImage?
    private var recognitionResult: String = ""
    private var messages = [String]()
    var allLowercase = false
    
    var word2VecProvider: W2VDistance? {
        didSet {
            print(NSDate(), "Finished loading data...")
        }
    }
    
    @IBAction func recordPressed(_ sender: Any) {
        if VoiceRecognizer.shared.isListening {
            VoiceRecognizer.shared.stopListening()
            recordButton.isEnabled = false
            recordButton.setTitle("Stopping", for: .disabled)
        } else {
            VoiceRecognizer.shared.startListening(gotResultBlock: gotNewWord, endBlock: recognitionEnded)
            recordButton.setTitle("Stop listening", for: [])
        }
    }
    
    // MARK: - Main Business Logic
    private func recognitionEnded(error: Error?) {
        if let error = error {
            let alert = UIAlertController(title: "Error",
                                          message: error.localizedDescription,
                                          preferredStyle: .alert)
            let cancelAction = UIAlertAction(title: "OK",
                                             style: .cancel, handler: nil)
            
            alert.addAction(cancelAction)
            self.present(alert, animated: true, completion: nil)
            return
        }
        recordButton.isEnabled = true
        recordButton.setTitle("Listen", for: [])
        let result = self.recognitionResult
        
        let words: [String]
        if allLowercase {
            words = result.split(separator: " ").map(String.init).map{$0.lowercased()}
        } else {
            words = NLPPreprocessor.preprocess(inputString: result) { error in
                messages.append(result)
                messages.append("This doesn't look like English.")
                reloadTable()
            }
        }
        
        let wordCount = words.count
        
        var stringToPassToW2V: String
        var stringToShowInUI: String
        switch wordCount {
        case 1:
            stringToPassToW2V = String(words.last!)
            stringToShowInUI = String(words.last!)
        case 2:
            let wordPair = Array(words.suffix(2))
            stringToPassToW2V = "\(wordPair[0]) \(wordPair[1])"
            stringToShowInUI = "\(wordPair[0]) ➖ \(wordPair[1])"
        case 3...:
            let wordTriplet = Array(words.suffix(3))
            stringToPassToW2V = "\(wordTriplet[0]) \(wordTriplet[1]) \(wordTriplet[2])"
            stringToShowInUI = "\(wordTriplet[0]) ➖ \(wordTriplet[1]) ➕ \(wordTriplet[2])"
        default:
            print("Warning: wrong number of input words.")
            return
        }
        print(stringToPassToW2V)
        messages.append(stringToShowInUI)
        reloadTable()
        
        DispatchQueue.main.async() { [weak self] in
            guard let `self` = self else { return }
            var response: String?
            if wordCount > 1 {
                response = self.getW2VAnalogy(sentence: stringToPassToW2V)?.capitalized
            } else {
                response = self.getW2VWord(word: stringToPassToW2V)?.capitalized
            }
            
            if response?.isEmpty ?? true || response == "``" {
                response = "I don't know this word."
            }
            
//            SpeechSynthesizer.shared.speakAloud(word: response!)
            self.messages.append(response!)
            self.reloadTable()
            
            print(response!)
        }
    }
    
    private func gotNewWord(string: String) {
        recognitionResult = string
    }
    
    // MARK: - Word2Vec
    private func getW2VAnalogy(sentence: String) -> String? {
        guard let words = word2VecProvider?.analogy(toPhrase: sentence, numberOfClosest: 1)?.keys else {
            return nil
        }
        return Array(words).last
    }
    
    private func getW2VWord(word: String) -> String? {
        guard let words = word2VecProvider?.closest(toWord: word, numberOfClosest: 1)?.keys else {
            return nil
        }
        // Uncomment for more random answers
        //        let randomIndex = Int(arc4random_uniform(UInt32(words.count)))
        //        return Array(words)[randomIndex]
        return Array(words).last
    }
    
    // MARK: - UI
    private func reloadTable() {
        tableView.reloadData()
        
        // First figure out how many sections there are
        let lastSectionIndex = tableView.numberOfSections - 1
        
        // Then grab the number of rows in the last section
        let lastRowIndex = tableView.numberOfRows(inSection: lastSectionIndex) - 1
        
        // Now just construct the index path
        let pathToLastRow = NSIndexPath(row: lastRowIndex, section: lastSectionIndex)
        
        // Make the last row visible
        tableView.scrollToRow(at: pathToLastRow as IndexPath, at: UITableViewScrollPosition.none, animated: true)
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        title = "Associations"
//        SpeechSynthesizer.shared.prepare()
    }
    
    // MARK: - UITableViewDelegate, UITableViewDataSource
    
    func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        
        let isOutgoing = indexPath.row%2 == 0
        if isOutgoing {
            guard let cell = tableView.dequeueReusableCell(withIdentifier: "my_message") as? OutgoingChatCell else {
                fatalError()
            }
            cell.message.text = messages[indexPath.row]
            cell.avatar.image = #imageLiteral(resourceName: "bot")
            cell.avatar.layer.cornerRadius = 20
            return cell
            
        } else {
            guard let cell = tableView.dequeueReusableCell(withIdentifier: "his_message") as? IncomingChatCell else {
                fatalError()
            }
            cell.message.text = messages[indexPath.row]
            cell.avatar.image = avatar
            cell.avatar.layer.cornerRadius = 20
            return cell
        }
    }
    
    func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        return messages.count
    }
    
    func tableView(_ tableView: UITableView, didSelectRowAt indexPath: IndexPath) {
        let term = messages[indexPath.row]
        if UIReferenceLibraryViewController.dictionaryHasDefinition(forTerm: term) {
            let referenceController = UIReferenceLibraryViewController(term: term)
            self.navigationController?.pushViewController(referenceController, animated: true)
        }
    }
}
