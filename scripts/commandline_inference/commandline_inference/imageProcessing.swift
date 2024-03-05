//
//  imageProcessing.swift
//  commandline_inference
//
//  Created by Paul Legner on 24.01.24.
//

import Foundation
import Cocoa
import CoreGraphics
import Vision
import UniformTypeIdentifiers

let rawJointNames: [String] = ["left_ear_joint",
                               "left_eye_joint",
                               "head_joint",
                               "right_eye_joint",
                               "right_ear_joint",
                               "left_hand_joint",
                               "left_forearm_joint",
                               "left_shoulder_1_joint",
                               "neck_1_joint",
                               "right_shoulder_1_joint",
                               "right_forearm_joint",
                               "right_hand_joint",
                               "left_upLeg_joint",
                               "left_leg_joint",
                               "left_foot_joint",
                               "root",
                               "right_upLeg_joint",
                               "right_leg_joint",
                               "right_foot_joint"];


func cgImageFromPath(filePath: String) -> CGImage {
    guard let image = NSImage(contentsOfFile: filePath) else {
        print ("Bad image path");
        exit(0);
    }
    guard let cgIm: CGImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
        print("Conversion error");
        exit(0);
    }
    
    print("width: \(cgIm.width), height: \(cgIm.height)");
    return cgIm;
}

func performHBPRequest( on image: CGImage) -> [VNHumanBodyPoseObservation] {
    var observations: [VNHumanBodyPoseObservation] = [];
    let requestHandler = VNImageRequestHandler(cgImage: image);
    let request = VNDetectHumanBodyPoseRequest(completionHandler: bodyPoseHandler);
    
    do {
        // Perform the body pose-detection request.
        try requestHandler.perform([request])
    } catch {
        print("Unable to perform the request: \(error).")
    }
    
    func bodyPoseHandler(request: VNRequest, error: Error?) {
        guard let results =
                request.results as? [VNHumanBodyPoseObservation] else {
            return
        }
        
        observations = results;
    }
    
    return observations;
}

@available(macOS 14.0, *)
func perform3DHBPRequest( on image: CGImage) -> [VNHumanBodyPose3DObservation] {
    var observations: [VNHumanBodyPose3DObservation] = [];
    let requestHandler = VNImageRequestHandler(cgImage: image);
    let request = VNDetectHumanBodyPose3DRequest(completionHandler: bodyPoseHandler);
    
    do {
        // Perform the body pose-detection request.
        try requestHandler.perform([request])
    } catch {
        print("Unable to perform the request: \(error).")
    }
    
    func bodyPoseHandler(request: VNRequest, error: Error?) {
        guard let results =
                request.results as? [VNHumanBodyPose3DObservation] else {
            print("here")
            return
        }
        
        observations = results;
    }
    
    return observations;
}

@available(macOS 14.0, *)
func drawSkeletonFrom3DPose(on image: CGImage, from observations: [VNHumanBodyPose3DObservation], writeTo destPath: String) {
    var groupedImagePoints: [[CGPoint]] = [];
    let orderedJointsGroupNames: [[VNHumanBodyPose3DObservation.JointName]] = [
        [.centerHead, .topHead],                                                                    // face
        [.centerShoulder, .rightShoulder, .spine, .rightHip, .root, .leftHip, .leftShoulder],       // torso
        [.leftWrist, .leftElbow, .leftShoulder],                                                    // left arm
        [.rightWrist, .rightElbow, .rightShoulder],                                                 // right arm
        [.leftAnkle, .leftKnee, .leftHip],                                                          // left leg
        [.rightAnkle, .rightKnee, .rightHip]                                                        // right leg
    ];
    
    // Extract keypoints
    let observation = observations[0];
    
    // Group by joint group and translate to image coordinates
    for jointsGroup in orderedJointsGroupNames {
        let recognizedJointsGroupCoords: [CGPoint] = jointsGroup.compactMap {
            guard let point = try? observation.pointInImage($0) else {
                return nil;
            }
            return VNImagePointForNormalizedPoint(point.location,
                                                  Int(image.width),
                                                  Int(image.height));
        }
        groupedImagePoints.append(recognizedJointsGroupCoords);
    }
    
    drawSkeleton(on: image, groupedImagePoints: groupedImagePoints, writeTo: destPath);
}

func drawSkeletonFrom2DPose(on image: CGImage, from observations: [VNHumanBodyPoseObservation], writeTo destPath: String) {
    var groupedImagePoints: [[CGPoint]] = [];
    let orderedJointsGroupNames: [[VNHumanBodyPoseObservation.JointName]] = [
        [.leftEar, .leftEye, .nose, .rightEye, .rightEar],                      // face
        [.neck, .rightShoulder, .rightHip, .root, .leftHip, .leftShoulder],     // torso
        [.leftWrist, .leftElbow, .leftShoulder],                                // left arm
        [.rightWrist, .rightElbow, .rightShoulder],                             // right arm
        [.leftAnkle, .leftKnee, .leftHip],                                      // left leg
        [.rightAnkle, .rightKnee, .rightHip]                                    // right leg
    ];
    
    // Extract keypoints
    let observation = observations[0];
    guard let recognizedPoints =
            try? observation.recognizedPoints(.all) else { return; }
    
    // Group by joint group and translate to image coordinates
    for jointsGroup in orderedJointsGroupNames {
        let recognizedJointsGroupCoords: [CGPoint] = jointsGroup.compactMap {
            guard let point = recognizedPoints[$0], point.confidence > 0 else {
                return nil;
            }
            return VNImagePointForNormalizedPoint(point.location,
                                                  Int(image.width),
                                                  Int(image.height));
        }
        groupedImagePoints.append(recognizedJointsGroupCoords);
    }
    
    drawSkeleton(on: image, groupedImagePoints: groupedImagePoints, writeTo: destPath);
}

func drawSkeleton(on image: CGImage, groupedImagePoints: [[CGPoint]], writeTo destPath: String) {
    // Create context and draw image and keypoints
    guard let context = CGContext(data: nil, width: image.width, height: image.height, bitsPerComponent: image.bitsPerComponent, bytesPerRow: image.bytesPerRow, space: image.colorSpace!, bitmapInfo: image.bitmapInfo.rawValue) else {
        print("Error creating context.");
        return;
    }
    context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height));
    context.setStrokeColor(CGColor(red: 0, green: 0,blue: 1,alpha: 1));
    context.setLineWidth(5);
        
    for imagePointGroup in groupedImagePoints {
        if (imagePointGroup.count < 2) { continue; }
        for i in 0...imagePointGroup.count-2 {
            context.move(to: imagePointGroup[i]);
            context.addLine(to: imagePointGroup[i+1]);
            context.strokePath();
        }
    }
    
    // Write context image to destination file
    print(destPath)
    let url = URL(fileURLWithPath: destPath);
    guard let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.jpeg.identifier as CFString, 1, nil) else {
        print("Error creating destination.");
        return;
    }

    guard let newImage = context.makeImage() else {print("Error making image"); return;};
    CGImageDestinationAddImage(destination, newImage, nil);
    guard CGImageDestinationFinalize(destination) else {
        print("Error finalizing CGImageDestination");
        return;
        }

    print("Modified image saved to \(destPath)");
}

func writeToJSON(observations: [VNHumanBodyPoseObservation], to destPath: String) {
    var pointDict: [String:[Double]] = [:];

    // Extract keypoints
    let observation = observations[0];
    guard let recognizedPoints =
            try? observation.recognizedPoints(.all) else { return; }
    
    // keypoints -> dict
    recognizedPoints.forEach {(key, value) in
        pointDict[key.rawValue.rawValue] = [value.x, value.y];
    }
    
    guard let jsonData = try? JSONSerialization.data(withJSONObject: pointDict, options: .prettyPrinted) else {
        print("JSON conversion failed.");
        return;
    }
    
    // Write to destination file
    let url = URL(fileURLWithPath: destPath);
    do {
        try jsonData.write(to: url);
    } catch {
        print("Error writing JSON data to file at \(destPath).");
        return;
    }
    
    print("Keypoints saved to JSON file \(destPath)");
}

func writeToCSV(observations: [VNHumanBodyPoseObservation], className: String, to destPath: String) {
    if(!FileManager.default.fileExists(atPath: destPath)) {
        var header = "class,";
        for (index, rawJointName) in rawJointNames.enumerated() {
            header += "\(rawJointName)_x,";
            header += "\(rawJointName)_y";
            if (index < rawJointNames.count-1) { header += ",";}
        }
        header += "\n";
        
        FileManager.default.createFile(atPath: destPath, contents: nil);
        print("Created file \(destPath).");
        
        do {
            try header.write(toFile: destPath, atomically: false, encoding: .utf8);
        } catch {
            print("Error while writing to file \(destPath).");
            return;
        }
    }
    
    // Extract keypoints
    let observation = observations[0];
    guard let recognizedPoints =
            try? observation.recognizedPoints(.all) else { return; }
    
    var rawRecognizedPoints: [String: VNRecognizedPoint] = [:];
    recognizedPoints.forEach {(key, value) in
        rawRecognizedPoints[key.rawValue.rawValue] = value;
    }
        
    guard let fileHandle = FileHandle(forWritingAtPath: destPath) else {
        print("Error creating file handle for \(destPath)");
        return;
    }
    
    // Write keypoint coordinates to string
    var keyPointString = "\(className),";
    for (index, rawJointName) in rawJointNames.enumerated() {
        if let point = rawRecognizedPoints[rawJointName] {
            if (point.confidence > 0) {
                let (x, y) = (point.x, point.y);
                keyPointString += "\(x),\(y)";
            } else {
                keyPointString += ",";
            }
        }
        if (index < rawJointNames.count-1) { keyPointString += ","; }
    }
    keyPointString += "\n";
    let data = keyPointString.data(using: .utf8)!;
    
    // Append string to file
    do {
        try fileHandle.seekToEnd();
        fileHandle.write(data);
        try fileHandle.close();
    } catch {
        print("Error while writing to file \(destPath) via file handle.");
    }
}
