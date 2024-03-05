//
//  ContentView.swift
//  VideoFitnessTracker
//
//  Created by Paul Legner on 06.02.24.
//

import SwiftUI


struct ContentView: View {
    var body: some View {
         PoseCaptureViewControllerRep().ignoresSafeArea()
    }
}


struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
