//
//  Date.swift
//  ConvNetSwift
//
//  Created by Alex Sosnovshchenko on 2/18/17.
//  Copyright © 2017 Alex Sosnovshchenko. All rights reserved.
//

import Foundation

/// Source: http://stackoverflow.com/a/27184261/2291058

public extension Date {
    /// Returns the amount of years from another date
    public func years(from date: Date) -> Int {
        return Calendar.current.dateComponents([.year], from: date, to: self).year ?? 0
    }
    /// Returns the amount of months from another date
    public func months(from date: Date) -> Int {
        return Calendar.current.dateComponents([.month], from: date, to: self).month ?? 0
    }
    /// Returns the amount of weeks from another date
    public func weeks(from date: Date) -> Int {
        return Calendar.current.dateComponents([.weekOfYear], from: date, to: self).weekOfYear ?? 0
    }
    /// Returns the amount of days from another date
    public func days(from date: Date) -> Int {
        return Calendar.current.dateComponents([.day], from: date, to: self).day ?? 0
    }
    /// Returns the amount of hours from another date
    public func hours(from date: Date) -> Int {
        return Calendar.current.dateComponents([.hour], from: date, to: self).hour ?? 0
    }
    /// Returns the amount of minutes from another date
    public func minutes(from date: Date) -> Int {
        return Calendar.current.dateComponents([.minute], from: date, to: self).minute ?? 0
    }
    /// Returns the amount of seconds from another date
    public func seconds(from date: Date) -> Int {
        return Calendar.current.dateComponents([.second], from: date, to: self).second ?? 0
    }
    /// Returns the amount of nanoseconds from another date
    public func nanoseconds(from date: Date) -> Int {
        return Calendar.current.dateComponents([.nanosecond], from: date, to: self).nanosecond ?? 0
    }
    /// Returns the a custom time interval description from another date
    public func offset(from date: Date) -> String {
        if years(from: date)        > 0 { return "\(years(from: date))y"   }
        if months(from: date)       > 0 { return "\(months(from: date))M"  }
        if weeks(from: date)        > 0 { return "\(weeks(from: date))w"   }
        if days(from: date)         > 0 { return "\(days(from: date))d"    }
        if hours(from: date)        > 0 { return "\(hours(from: date))h"   }
        if minutes(from: date)      > 0 { return "\(minutes(from: date))m" }
        if seconds(from: date)      > 0 { return "\(seconds(from: date))s" }
        if nanoseconds(from: date)  > 0 { return "\(nanoseconds(from: date))ns" }
        return ""
    }
}
