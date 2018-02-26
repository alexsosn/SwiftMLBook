//
//  ChatCell.swift
//  VectorBot
//
//  Created by Oleksandr on 6/22/17.
//  Copyright © 2017 OWL. All rights reserved.
//

import Foundation
import UIKit

class IncomingChatCell: UITableViewCell {
    @IBOutlet weak var avatar: UIImageView!
    @IBOutlet weak var message: UILabel!
    
}

class OutgoingChatCell: UITableViewCell {
    @IBOutlet weak var avatar: UIImageView!
    @IBOutlet weak var message: UILabel!
}
