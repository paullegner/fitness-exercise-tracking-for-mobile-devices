# Fitness Exercise Tracking For Mobile Devices

This is the accompanying repository to my Bachelors thesis on the topic of **Fitness Exercise Tracking for Mobile Devices Based on Video Analysis**. 

## Contents

The repository includes three directories, which are detailed in the following.

### Scripts

This directory includes any Python utilty scripts used for building our dataset and training our classifiers. Notably: 
* **blaze_pose_inference.py**: Runs a set of images through a BlazePose model, calculates additional features and outputs the results to a csv file or draws them on an image
* **exercise_classifier.py**: Trains a specified Sklearn classifier on a given train (and test) dataset, evaluates its performance and converts it to a coremlmodel
* **commandline_inference**: MacOS command line tool to run a Vision pose request on set of given images. Results are output to csv or image.
* other utilities 

### VideoFitnessTracker 

This is the project directory for the VideoFitnessTracker application which serves as a demonstration for our processing pipeline. 
The individual .swift source code files are located in the subdirectory of the same name. To deploy the application via Xcode, open the project via the **VideoFitnessTracker.xcworkspace** file.

### Models

This directory contains the classification models used by the VideoFitnessTracker application converted to coremlmodel format and pickled as joblib files.
