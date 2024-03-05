//
//  RingBuffer.swift
//  VideoFitnessTracker
//
//  Created by Paul Legner on 28.02.24.
//

import Foundation

struct RingBuffer<T: Hashable> {
    private var buffer: [T];
    private var size: Int;
    
    public init(bufferSize: Int) {
        size = bufferSize;
        buffer = [];
    }
    
    public func getCapacity() -> Int {
        return buffer.count;
    }
    
    public func getLatest() -> T {
        return buffer[0];
    }
    
    public func mostFrequent() -> T {
        var counts: [T: Int] = [:]
        for item in buffer {
            counts[item] = (counts[item] ?? 0) + 1;
        }
        let mostFrequent = counts.max { a, b in a.value < b.value }
        return mostFrequent!.key;
    }
    
    public mutating func push(_ item: T) {
        if buffer.count >= size {
            _ = buffer.popLast();
        }
        buffer.insert(item, at: 0);
    }
    
    public func isOnly(item: T) -> Bool {
        return buffer.filter{$0 == item}.count == buffer.count;
    }
}
