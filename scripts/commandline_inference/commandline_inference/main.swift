//
//  main.swift
//  commandline_inference
//
//  Created by Paul Legner on 23.01.24.
//

import Foundation
import Vision

let exercise_type: String = "push-up";
//let sourcePath: String = "/Users/paul1/Documents/Uni/23WS_IMI/BA/BA_2023/action_classifier_dataset/train/plank/plank_2/frame19.jpg";
//let source_dir: String = "/Users/paullegner/Documents/Uni/23WS_IMI/BA/training_data/\(exercise_type)"
let source_dir = "/Users/paullegner/Documents/Uni/23WS_IMI/BA/training_data/random_set";
let destinationPath: String = "/Users/paullegner/Desktop/test_keypoints.csv";

if #available(macOS 14.0, *) {
    let clock = ContinuousClock();

    guard let dictEnumerator = FileManager.default.enumerator(atPath: source_dir) else {
        print("Directory Enumerator could not be initialized.");
        exit(0);
    };
    
    var no_detection_files: [String] = [];
    var total_inference_time = 0.0;
    var total_inferences = 0;
    
    while let elem = dictEnumerator.nextObject() as? String {
        print(elem);
        if(elem.contains("frame")) {
            print("Processing file \(elem).");
            let image = cgImageFromPath(filePath: source_dir + "/" + elem);
            var results: [VNHumanBodyPose3DObservation] = [];
            let time = clock.measure({
                results = perform3DHBPRequest(on: image);
            });
            total_inference_time += Double(time.components.attoseconds) * 0.000000000000000001;
            total_inferences += 1;
            if (results.isEmpty) {
                print("No keypoints detected.");
                no_detection_files.append(elem);
            } else {
                //drawSkeletonFrom3DPose(on: image, from: results, writeTo: "/Users/paullegner/Desktop/results/\(elem)");
                //writeToJSON(observations: results, to: destinationPath);
                //writeToCSV(observations: results, className: exercise_type, to: destinationPath);
            }
        }
    }
    print(total_inference_time / Double(total_inferences))
    print(total_inferences)
}



