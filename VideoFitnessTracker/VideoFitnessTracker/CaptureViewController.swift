//
//  ViewController.swift
//  VideoFitnessTracker
//
//  Created by Paul Legner on 06.02.24.
//

import SwiftUI
import UIKit
import AVFoundation
import Vision

// This class was partly adopted from the Apple Developer Documentation example on live object detection, available at:
// https://developer.apple.com/documentation/vision/recognizing_objects_in_live_capture
class CaptureViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
    
    var bufferSize: CGSize = .zero;
    var rootLayer: CALayer! = nil;
    var currentPosition: AVCaptureDevice.Position = .unspecified;
    
    private let session = AVCaptureSession();
    private var previewLayer: AVCaptureVideoPreviewLayer! = nil;
    private let videoDataOutput = AVCaptureVideoDataOutput();
    private let videoDataOutputQueue = DispatchQueue(label: "VideoDataOutput", qos: .userInitiated, attributes: [], autoreleaseFrequency: .workItem);
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        // to be implemented in the subclass
    }
    
    override func viewDidLoad() {
        super.viewDidLoad();
        setupAVCapture();
    }
    
    func setupAVCapture() {
        var deviceInput: AVCaptureDeviceInput!;
        
        // Select a video device, make an input
        let videoDevice = AVCaptureDevice.DiscoverySession(deviceTypes: [.builtInWideAngleCamera], mediaType: .video, position: .front).devices.first;
        do {
            deviceInput = try AVCaptureDeviceInput(device: videoDevice!);
            currentPosition = videoDevice!.position;
        } catch let error {
            print("Could not create video device input: \(error)");
            return;
        }
        
        session.beginConfiguration();
        
        // Add a video input
        guard session.canAddInput(deviceInput) else {
            print("Could not add video device input to the session");
            session.commitConfiguration();
            return;
        }
        session.addInput(deviceInput)
        if session.canAddOutput(videoDataOutput) {
            session.addOutput(videoDataOutput);
            // Add a video data output
            videoDataOutput.alwaysDiscardsLateVideoFrames = true;
            videoDataOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_420YpCbCr8BiPlanarFullRange)];
            videoDataOutput.setSampleBufferDelegate(self, queue: videoDataOutputQueue);
        } else {
            print("Could not add video data output to the session");
            session.commitConfiguration();
            return;
        }
        let captureConnection = videoDataOutput.connection(with: .video);
        // Always process the frames
        captureConnection?.isEnabled = true;
        do {
            try  videoDevice!.lockForConfiguration();
            let dimensions = CMVideoFormatDescriptionGetDimensions((videoDevice?.activeFormat.formatDescription)!);
            bufferSize.width = CGFloat(dimensions.width);
            bufferSize.height = CGFloat(dimensions.height);
            videoDevice!.unlockForConfiguration();
        } catch {
            print(error);
        }
        session.commitConfiguration();
        previewLayer = AVCaptureVideoPreviewLayer(session: session);
        previewLayer.videoGravity = AVLayerVideoGravity.resizeAspectFill;
        rootLayer = view.layer;
        previewLayer.frame = rootLayer.bounds;
        rootLayer.addSublayer(previewLayer);
    }
    
    func startCaptureSession() {
        DispatchQueue.global(qos: .background).async(execute: {
            self.session.startRunning();
        })
    }
}
