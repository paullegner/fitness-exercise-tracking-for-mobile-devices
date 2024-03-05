//
//  PoseProcessor.swift
//  VideoFitnessTracker
//
//  Created by Paul Legner on 01.03.24.
//

import Foundation
import CoreML
import MLKit
import simd


// subset of COCO keypoint labels, for classifier input
let cocoKeypoints: [PoseLandmarkType] = [
    .nose, .leftEye, .rightEye, .leftEar, .rightEar,
    .leftShoulder, .rightShoulder, .leftElbow, .rightElbow, .leftWrist, .rightWrist,
    .leftHip, .rightHip, .leftKnee, .rightKnee, .leftAnkle, .rightAnkle
];

// joint labels grouped by angles, for classifier input
let relevantAngles: KeyValuePairs<String, ((PoseLandmarkType, PoseLandmarkType), (PoseLandmarkType, PoseLandmarkType))> = [
    "left_shoulder_angle": ((.leftElbow, .leftShoulder), (.leftHip, .leftShoulder)),                            // left upper arm, left torso
    "right_shoulder_angle": ((.rightElbow, .rightShoulder), (.rightHip, .rightShoulder)),                       // right upper arm, right torso
    "left_inner_shoulder_angle": ((.leftShoulder, .rightShoulder), (.leftElbow, .leftShoulder)),                // collarbone, left upper arm
    "right_inner_shoulder_angle": ((.leftShoulder, .rightShoulder), (.rightElbow, .rightShoulder)),             // collarbone, right upper arm
    "left_elbow_angle": ((.leftElbow, .leftShoulder), (.leftWrist, .leftElbow)),                                // left upper arm, left forearm
    "right_elbow_angle": ((.rightElbow, .rightShoulder), (.rightWrist, .rightElbow)),                           // right upper arm, right forearm
    "left_hip_angle": ((.leftHip, .leftShoulder), (.leftKnee, .leftHip)),                                       // left torso, left upper leg
    "right_hip_angle": ((.rightHip, .rightShoulder), (.rightKnee, .rightHip)),                                  // right torso, right upper leg
    "left_inner_hip_angle": ((.leftHip, .rightHip), (.leftKnee, .leftHip)),                                     // hip, left upper leg
    "right_inner_hip_angle": ((.leftHip, .rightHip), (.rightKnee, .rightHip)),                                  // hip, right upper leg
    "left_knee_angle": ((.leftKnee, .leftHip), (.leftAnkle, .leftKnee)),                                        // left upper leg, left lower leg
    "right_knee_angle": ((.rightKnee, .rightHip), (.rightAnkle, .rightKnee)),                                   // right upper leg, right lower leg
]

struct PoseProcessor {
    private var exerciseClassifier: svc_angles_classifier? = nil;
    private var stageClassifier_pullup: svc_angles_pullup_stage_classifier? = nil;
    private var stageClassifier_pushup: svc_angles_pushup_stage_classifier? = nil;
    private var stageClassifier_squat: svc_angles_squat_stage_classifier? = nil;
    
    private var exerciseBuffer: RingBuffer<String>? = nil;
    private var currentExercise: String = "";
    private var stageBuffer: RingBuffer<String>? = nil;
    private var currStage: String = "";
    private var repCount: Int = 0;
    
    public init() {
        do {
            exerciseClassifier = try svc_angles_classifier(configuration: MLModelConfiguration());
            stageClassifier_pullup = try svc_angles_pullup_stage_classifier(configuration: MLModelConfiguration());
            stageClassifier_pushup = try svc_angles_pushup_stage_classifier(configuration: MLModelConfiguration());
            stageClassifier_squat = try svc_angles_squat_stage_classifier(configuration: MLModelConfiguration());
            exerciseBuffer = RingBuffer(bufferSize: 120);
            stageBuffer = RingBuffer(bufferSize: 5);
        } catch let error {
            print("Could not initialize PoseProcessor");
            print(error);
            return;
        }
    }
    
    public mutating func processPose(_ pose: Pose) -> (String, Int)? {
        let angles = self.calculateAngles(pose);
        //let normalizedCoords = self.normalizePose(pose);
        
        if let input = try? MLMultiArray(angles) {
            do {
                let exercisePrediction = try self.exerciseClassifier?.prediction(pose_angles: input);
                let predictedExercise = exercisePrediction?.predicted_exercise ?? "";
                
                self.exerciseBuffer!.push(predictedExercise);
                if self.exerciseBuffer!.mostFrequent() == predictedExercise {
                    let predictedStage: String = self.predictStage(from: input, for: predictedExercise) ?? "";
                    self.stageBuffer?.push(predictedStage);
                    // predicted exercise is confirmed
                    if self.currentExercise == predictedExercise {
                        if self.stageBuffer!.isOnly(item: predictedStage) {
                            // predicted stage confirmed
                            if self.currStage != predictedStage {
                                //update repetition counter
                                if predictedStage == "start" {
                                    self.repCount += 1;
                                }
                                self.currStage = predictedStage;
                            }
                        }
                    } else {
                        self.repCount = 0;
                        self.currentExercise = predictedExercise;
                    }
                }
                print("\(self.currentExercise)  -> \(self.repCount)");
                return (self.currentExercise, self.repCount);
            } catch let error {
                print(error);
                return nil;
            }
        } else {
            print("Could not construct feature array");
            return nil;
        }
    }
    
    private func predictStage(from input: MLMultiArray, for exerciseType: String) -> String? {
        var predictedStage: String = "";
        do {
            if exerciseType == "push-up" {
                let stagePrediction = try self.stageClassifier_pushup?.prediction(pose_angles: input);
                predictedStage = stagePrediction!.predicted_stage;
            } else if exerciseType == "pull-up" {
                let stagePrediction = try self.stageClassifier_pullup?.prediction(pose_angles: input);
                predictedStage = stagePrediction!.predicted_stage;
            } else if exerciseType == "squat" {
                let stagePrediction = try self.stageClassifier_squat?.prediction(pose_angles: input);
                predictedStage = stagePrediction!.predicted_stage;
            } else {
                print("No classifier for exercise type \(exerciseType) available.");
                return nil;
            }
        } catch let error {
            print("Error while predicting stage.");
            print(error);
            return nil;
        }
        
        return predictedStage;
    }
    
    private func calculateAngles(_ pose: Pose) -> [Double] {
        var angleValues: [Double] = [];
        for (_, keypoints) in relevantAngles {
            let limb1 = extractCoordinates(from: pose.landmark(ofType: keypoints.0.0)) - extractCoordinates(from: pose.landmark(ofType: keypoints.0.1));
            let limb2 = extractCoordinates(from: pose.landmark(ofType: keypoints.1.0)) - extractCoordinates(from: pose.landmark(ofType: keypoints.1.1));
            
            let angle = acos(dot(limb1, limb2)/(length(limb1) * length(limb2)));
            angleValues.append(angle);
        }
        return angleValues
    }
    
    private func extractCoordinates(from landmark: PoseLandmark) -> SIMD3<Double> {
        return simd_double3(landmark.position.x, landmark.position.y, landmark.position.z)
    }
    
    private func normalizePose(_ pose: Pose) -> [Double] {
        var xMin = Double.infinity
        var xMax = 0.0
        var yMin = Double.infinity
        var yMax = 0.0
        var zMin = Double.infinity
        var zMax = -Double.infinity
        var jointCoordinates: [Double] = [];
        
        for landmarkName in cocoKeypoints {
            let landmark = pose.landmark(ofType: landmarkName);
            xMax = Double.maximum(xMax, landmark.position.x)
            xMin = Double.minimum(xMin, landmark.position.x)
            yMax = Double.maximum(yMax, landmark.position.y)
            yMin = Double.minimum(yMin, landmark.position.y)
            zMax = Double.maximum(zMax, landmark.position.z)
            zMin = Double.minimum(zMin, landmark.position.z)
        }
        for landmarkName in cocoKeypoints {
            let landmark = pose.landmark(ofType: landmarkName);
            let x = (landmark.position.x - xMin) / (xMax - xMin)
            let y = (landmark.position.y - yMin) / (yMax - yMin)
            let z = (landmark.position.z - zMin) / (zMax - zMin)
            jointCoordinates.append(contentsOf: [x, y, z]);
        }
        
        //jointCoordinates = rotatePose(coordinates: jointCoordinates, center: (xMax-xMin, yMax-yMin));
        return jointCoordinates;
    }
    
    private func rotatePose(coordinates: [Double], center: (Double, Double)) -> [Double] {
        var result: [Double] = [];
        for i in stride(from: 0, to: coordinates.count - 1, by: 3) {
            var x = coordinates[i];
            var y = coordinates[i+1];
            let z = coordinates[i+2];
            x -= center.0;
            y -= center.1;
            x = y + center.0;
            y = -x + center.1;
            result.append(contentsOf: [x, y, z]);
        }
        return result;
    }
}
