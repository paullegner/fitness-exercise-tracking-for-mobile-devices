//
//  Detector.swift
//  VideoFitnessTracker
//
//  Created by Paul Legner on 18.02.24.
//

import SwiftUI
import UIKit
import AVFoundation
import Vision
import MLKit
import simd


// for drawing purposes
let groupedJointNamess: [[PoseLandmarkType]] = [
    [.leftEar, .leftEye, .nose, .rightEye, .rightEar],                      // face
    [.rightShoulder, .rightHip, .leftHip, .leftShoulder],                   // torso
    [.leftWrist, .leftElbow, .leftShoulder],                                // left arm
    [.rightWrist, .rightElbow, .rightShoulder],                             // right arm
    [.leftAnkle, .leftKnee, .leftHip],                                      // left leg
    [.rightAnkle, .rightKnee, .rightHip]                                    // right leg
];
let drawingColors: [CGColor] = [
    CGColor(red: 0.98, green: 0.14, blue: 0.45, alpha: 1.0),
    CGColor(red: 0.65, green: 0.89, blue: 0.18, alpha: 1.0),
    CGColor(red: 0.4, green: 0.85, blue: 0.94, alpha: 1.0),
    CGColor(red: 0.99, green: 0.59, blue: 0.12, alpha: 1.0),
    CGColor(red: 0.68, green: 0.51, blue: 1.0, alpha: 1.0),
    CGColor(red: 0.98, green: 0.15, blue: 0.45, alpha: 1.0),
];


// This class was partly adopted from the Apple Developer Documentation example on live object detection, available at:
// https://developer.apple.com/documentation/vision/recognizing_objects_in_live_capture
class PoseCaptureViewController: CaptureViewController {
    private var poseOverlay: CALayer! = nil;
    private var informationOverlay: CAShapeLayer! = nil;
    private var poseDetector: PoseDetector? = nil;
    private var poseProcessor: PoseProcessor? = nil;
    
    override func setupAVCapture() {
        super.setupAVCapture();
        setupLayers();
        updateLayerGeometry();
        setupPoseDetection();
        
        startCaptureSession();
    }
    
    func setupLayers() {
        // layer canvas for pose skeleton
        poseOverlay = CALayer();
        poseOverlay.name = "poseOverlay";
        poseOverlay.bounds = CGRect(x: 0.0,
                                    y: 0.0,
                                    width: bufferSize.width,
                                    height: bufferSize.height);
        poseOverlay.position = CGPoint(x: rootLayer.bounds.midX, y: rootLayer.bounds.midY);
        
        // layer canvas for exercise and rep count information
        informationOverlay = CAShapeLayer();
        informationOverlay.name = "informationOverlay";
        informationOverlay.bounds = rootLayer.bounds;
        informationOverlay.position = CGPoint(x: rootLayer.bounds.midX, y: rootLayer.bounds.midY);
        informationOverlay.path = UIBezierPath(rect: CGRect(x: 0.0, y: rootLayer.bounds.height - 200.0, width: rootLayer.bounds.width, height: 200.0)).cgPath;
        informationOverlay.fillColor = UIColor(white: 0.0, alpha: 0.75).cgColor;
        
        // exercise and rep count labels
        let exreciseLabelLayer = CATextLayer();
        exreciseLabelLayer.frame = CGRect(x: 10.0, y: rootLayer.bounds.height - 200.0 + 10.0, width: rootLayer.bounds.width, height: 200.0);
        exreciseLabelLayer.font = UIFont.boldSystemFont(ofSize: 25);
        exreciseLabelLayer.fontSize = 25;
        exreciseLabelLayer.alignmentMode = .left;
        exreciseLabelLayer.string = "Exercise:";
        exreciseLabelLayer.foregroundColor = UIColor(white: 0.8, alpha: 1.0).cgColor;
        
        let countLabelLayer = CATextLayer();
        countLabelLayer.frame = CGRect(x: 0.0, y: rootLayer.bounds.height - 200.0 + 10.0, width: rootLayer.bounds.width - 90.0, height: 200.0);
        countLabelLayer.font = UIFont.boldSystemFont(ofSize: 25);
        countLabelLayer.fontSize = 25;
        countLabelLayer.alignmentMode = .right;
        countLabelLayer.string = "Count:";
        countLabelLayer.foregroundColor = UIColor(white: 0.8, alpha: 1.0).cgColor;
        
        // append layers to view
        rootLayer.addSublayer(poseOverlay);
        rootLayer.addSublayer(informationOverlay);
        rootLayer.addSublayer(exreciseLabelLayer);
        rootLayer.addSublayer(countLabelLayer);
    }
    
    func setupPoseDetection() {
        let options = PoseDetectorOptions();
        options.detectorMode = .stream;
        poseDetector = PoseDetector.poseDetector(options: options);
        poseProcessor = PoseProcessor();
    }
    
    override func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        let image = VisionImage(buffer: sampleBuffer);
        image.orientation = self.imageOrientation(deviceOrientation: UIDevice.current.orientation, cameraPosition: self.currentPosition);
        var results: [Pose] = [];
        
        // Needs to be called from a background thread
        DispatchQueue.global(qos: .background).async(execute: {
            do {
                results = try self.poseDetector!.results(in: image);
                
                // Process pose data on main queue
                DispatchQueue.main.async(execute: {
                    self.poseOverlay.sublayers = nil;
                    if !results.isEmpty {
                        if let pose = results.first {
                            self.drawJoints(pose);
                            if let (currentExercise, repCount) = self.poseProcessor?.processPose(pose) {
                                self.buildInfoLayer(exercise: currentExercise, repetitions: repCount);
                            }
                        }
                    }
                })
            } catch let error {
                print("Error during pose detection");
                print(error);
                return;
            }
        })
    }
    
    func drawJoints(_ observation: Pose) {
        CATransaction.begin();
        CATransaction.setValue(kCFBooleanTrue, forKey: kCATransactionDisableActions);
        
        // Group by joint group and translate to image coordinates
        var imagePointGroups: [[CGPoint]] = [];
        for group in groupedJointNamess {
            let recognizedJointsGroupCoords: [CGPoint] = group.compactMap {
                let joint = observation.landmark(ofType: $0);
                if joint.inFrameLikelihood > 0.5 {
                    let position = joint.position;
                    return CGPoint(x:Int(position.x), y:  Int(position.y));
                } else {
                    return nil;
                }
            }
            imagePointGroups.append(recognizedJointsGroupCoords);
        }
        
        for (i, imagePointGroup) in imagePointGroups.enumerated() {
            if (imagePointGroup.count < 2) { continue; }
            let path = UIBezierPath();
            for j in 0...imagePointGroup.count-2 {
                // draw edge
                path.move(to: imagePointGroup[j]);
                path.addLine(to: imagePointGroup[j+1]);
            }
            // close torso
            if i == 1 {
                path.addLine(to: imagePointGroup[0]);
            }
            
            let drawingLayer = CAShapeLayer();
            drawingLayer.path = path.cgPath;
            drawingLayer.strokeColor = drawingColors[i];
            drawingLayer.fillColor = CGColor(gray: 1, alpha: 0);
            drawingLayer.lineWidth = 10;
            poseOverlay.addSublayer(drawingLayer);
        }
        
        self.updateLayerGeometry();
        CATransaction.commit();
    }
    
    func buildInfoLayer(exercise: String, repetitions: Int) {
        CATransaction.begin();
        CATransaction.setValue(kCFBooleanTrue, forKey: kCATransactionDisableActions);
        self.informationOverlay.sublayers = nil;
        
        let exerciseTextLayer = CATextLayer();
        exerciseTextLayer.frame = CGRect(x: 20.0, y: rootLayer.bounds.height - 200.0 + 85.0, width: rootLayer.bounds.width, height: 200.0);
        exerciseTextLayer.font = UIFont.boldSystemFont(ofSize: 35);
        exerciseTextLayer.fontSize = 35;
        exerciseTextLayer.alignmentMode = .left;
        exerciseTextLayer.string = exercise.uppercased();
        exerciseTextLayer.foregroundColor = UIColor(white: 1.0, alpha: 1.0).cgColor;
        informationOverlay.addSublayer(exerciseTextLayer);
        
        let stageTextLayer = CATextLayer();
        stageTextLayer.frame = CGRect(x: 0.0, y: rootLayer.bounds.height - 200.0 + 35.0, width: rootLayer.bounds.width - 20.0, height: 200.0);
        stageTextLayer.font = UIFont.boldSystemFont(ofSize: 35);
        stageTextLayer.fontSize = 120;
        stageTextLayer.alignmentMode = .right;
        stageTextLayer.string = String(format: "%02d", repetitions);
        stageTextLayer.foregroundColor = UIColor(white: 1.0, alpha: 1.0).cgColor;
        informationOverlay.addSublayer(stageTextLayer);
        
        CATransaction.commit();
    }
    
    func updateLayerGeometry() {
        let bounds = rootLayer.bounds;
        var scale: CGFloat;
        
        let xScale: CGFloat = bounds.size.width / bufferSize.height;
        let yScale: CGFloat = bounds.size.height / bufferSize.width;
        
        scale = fmax(xScale, yScale);
        if scale.isInfinite {
            scale = 1.0;
        }
        CATransaction.begin();
        CATransaction.setValue(kCFBooleanTrue, forKey: kCATransactionDisableActions);
        
        // rotate the layer into screen orientation and scale and mirror
        poseOverlay.setAffineTransform(CGAffineTransform(rotationAngle: CGFloat(.pi / 2.0)).scaledBy(x: scale, y: -scale));

        // center the layer
        poseOverlay.position = CGPoint(x: bounds.midX, y: bounds.midY);

        CATransaction.commit();
        
    }
    
    // taken from the MLKit Pose detection documentation, available at:
    // https://developers.google.com/ml-kit/vision/pose-detection/ios
    func imageOrientation(
      deviceOrientation: UIDeviceOrientation,
      cameraPosition: AVCaptureDevice.Position
    ) -> UIImage.Orientation {
        switch deviceOrientation {
        case .portrait:
            return cameraPosition == .front ? .leftMirrored : .right;
        case .landscapeLeft:
            return cameraPosition == .front ? .downMirrored : .up;
        case .portraitUpsideDown:
            return cameraPosition == .front ? .rightMirrored : .left;
        case .landscapeRight:
            return cameraPosition == .front ? .upMirrored : .down;
        default:
            return .up;
        }
    }
}


struct PoseCaptureViewControllerRep: UIViewControllerRepresentable {
    func updateUIViewController(_ uiViewController: UIViewController, context: Context) {
    }
    
    func makeUIViewController(context: Context) -> UIViewController {
        return PoseCaptureViewController();
    }
}
