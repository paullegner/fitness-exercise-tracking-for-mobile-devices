from collections import OrderedDict
import csv
import os
import sys
import time
import numpy as np
import cv2
from tqdm import tqdm
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2


model_path = '../../../BA/BA_2023/scripts/pose_landmarker_lite.task'
image_path = '../../../BA/training_data/push-up/push-up_8/frame3.jpg'

modes = ['training', 'test']
features = ['angles', 'points', 'normalized_points']
exercises = ['push-up', 'pull-up', 'squat']
stages = ['start', 'end']

coco_body_parts = OrderedDict([
	(0, 'nose'),
	(2, 'left_eye'),
	(5, 'right_eye'),
	(7, 'left_ear'),
	(8, 'right_ear'),
	(11, 'left_shoulder'),
	(12, 'right_shoulder'),
	(13, 'left_elbow'),
	(14, 'right_elbow'),
	(15, 'left_wrist'),
	(16, 'right_wrist'),
	(23, 'left_hip'),
	(24, 'right_hip'),
	(25, 'left_knee'),
	(26, 'right_knee'),
	(27, 'left_ankle'),
	(28, 'right_ankle'),
	])

groupedJoints = [
	[7, 2, 0, 5, 8],
	[11, 12, 24, 23, 11],
	[15, 13, 11],
	[16, 14, 12],
	[27, 25, 23],
	[28, 26, 24]
]

limbs = {
	'left_forearm': (15, 13),
	'right_forearm': (16, 14),
	'left_upper_arm': (13, 11),
	'right_upper_arm': (14, 12),
	'collarbone': (11, 12),
	'left_torso': (23, 11),
	'right_torso': (24, 12),
	'hip': (23, 24),
	'left_upper_leg': (25, 23),
	'right_upper_leg': (26, 24),
	'left_lower_leg': (27, 25),
	'right_lower_leg': (28, 26)
}

relevant_angles = OrderedDict([
	('left_shoulder_angle', (limbs['left_upper_arm'], limbs['left_torso'])), 
	('right_shoulder_angle', (limbs['right_upper_arm'], limbs['right_torso'])),
	('left_inner_shoulder_angle', (limbs['collarbone'], limbs['left_upper_arm'])),
	('right_inner_shoulder_angle', (limbs['collarbone'], limbs['right_upper_arm'])),
	('left_elbow_angle', (limbs['left_upper_arm'], limbs['left_forearm'])),
	('right_elbow_angle', (limbs['right_upper_arm'], limbs['right_forearm'])),
	('left_hip_angle', (limbs['left_torso'], limbs['left_upper_leg'])),
	('right_hip_angle', (limbs['right_torso'], limbs['right_upper_leg'])),
	('left_inner_hip_angle', (limbs['hip'], limbs['left_upper_leg'])),
	('right_inner_hip_angle', (limbs['hip'], limbs['right_upper_leg'])),
	('left_knee_angle', (limbs['left_upper_leg'], limbs['left_lower_leg'])),
	('right_knee_angle', (limbs['right_upper_leg'], limbs['right_lower_leg'])),
])

class BlazePoseInferencer():
	total_inference_time = 0.0
	total_inferences = 0
	landmarker = None

	def __init__(self):
		BaseOptions = mp.tasks.BaseOptions
		PoseLandmarker = mp.tasks.vision.PoseLandmarker
		PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
		VisionRunningMode = mp.tasks.vision.RunningMode

		options = PoseLandmarkerOptions(
			base_options=BaseOptions(model_asset_path=model_path),
			running_mode=VisionRunningMode.IMAGE)
		
		self.landmarker = PoseLandmarker.create_from_options(options)

	def run_inference_on(self, image, draw_pose):

		start = time.time()
		mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
		pose_landmarker_result = self.landmarker.detect(mp_image)
		self.total_inference_time += time.time() - start
		self.total_inferences += 1

		if (draw_pose):
			annotated_image = self.draw_coco_body(image, pose_landmarker_result)
			#annotated_image = self.draw_landmarks_on_image(image, pose_landmarker_result)
			cv2.imshow('ann', annotated_image)
			cv2.waitKey(0)
		
		return(pose_landmarker_result)

	def draw_coco_body(self, image, results):
		image_height, image_width, _ = image.shape
		if len(results.pose_landmarks):
			landmarks = results.pose_landmarks[0]
			for group in groupedJoints:
				for i in range(0, len(group)-1):
					landmark1 = landmarks[group[i]]
					landmark2 = landmarks[group[i+1]]
					x1, y1 = (image_width * landmark1.x, image_height * landmark1.y)
					x2, y2 = (image_width * landmark2.x, image_height * landmark2.y)
					cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 10)
			return image
		else:
			return None
         

	# function taken from Mediapipes pose_landmarker example, available at: 
	# https://github.com/googlesamples/mediapipe/tree/main/examples/pose_landmarker/python
	def draw_landmarks_on_image(self, rgb_image, detection_result):
		pose_landmarks_list = detection_result.pose_landmarks
		annotated_image = np.copy(rgb_image)

		# Loop through the detected poses to visualize.
		for idx in range(len(pose_landmarks_list)):
			pose_landmarks = pose_landmarks_list[idx]

			# Draw the pose landmarks.
			pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
			pose_landmarks_proto.landmark.extend([
			landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
			])
			solutions.drawing_utils.plot_landmarks(
			pose_landmarks_proto, 
			solutions.pose.POSE_CONNECTIONS, 
			landmark_drawing_spec = solutions.drawing_utils.DrawingSpec(color=(0, 0, 0), thickness=5),
            connection_drawing_spec = solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=5), elevation=10, azimuth=10)
			solutions.drawing_utils.draw_landmarks(
			annotated_image,
			pose_landmarks_proto,
			solutions.pose.POSE_CONNECTIONS,  
			landmark_drawing_spec = solutions.drawing_utils.DrawingSpec(color=(0, 0, 0), thickness=5),
            connection_drawing_spec = solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=5))
		return annotated_image
	
	def calculate_and_write_angles(self, pose, class_name):
		angles = [class_name]
		
		for associated_landmarks in relevant_angles.values():
			limb1_landmarks = associated_landmarks[0]
			limb2_landmarks = associated_landmarks[1]
			limb1 = self.extract_coords(pose[limb1_landmarks[0]]) - self.extract_coords(pose[limb1_landmarks[1]])
			limb2 = self.extract_coords(pose[limb2_landmarks[0]]) - self.extract_coords(pose[limb2_landmarks[1]])
			
			angle = np.arccos(
				np.dot(limb1, limb2) /
				(np.linalg.norm(limb1) * np.linalg.norm(limb2))
				)
			# angle = np.arctan2(np.linalg.norm(np.cross(limb1, limb2)), np.dot(limb1, limb2))

			angles.append(angle)
		
		with open(csv_path, 'a', newline='') as file:
			writer = csv.writer(file)
			writer.writerow(angles)

	def write_pose(self, pose, class_name, only_coco=True, normalize=False):
		coordinates = [class_name]
		if only_coco:
			pose = [pose[i] for i in coco_body_parts.keys()]
		if normalize:
			pose = self.normalizedPositions(pose)
		
		for landmark in pose:
				coordinates.extend([landmark.x, landmark.y, landmark.z])

		with open(csv_path, 'a', newline='') as file:
			writer = csv.writer(file)
			writer.writerow(coordinates)

	def extract_coords(self, landmark):
		return np.array([landmark.x, landmark.y, landmark.z])

	def normalizedPositions(self, pose):
		coords = [(landmark.x, landmark.y, landmark.z) for landmark in pose]
		x_list, y_list, z_list = zip(*coords)
		x_max = max(x_list)
		x_min = min(x_list)
		y_max = max(y_list)
		y_min = min(y_list)
		z_max = max(z_list)
		z_min = min(z_list)

		for landmark in pose:
			landmark.x = (landmark.x - x_min) / (x_max-x_min)				# Normalize to 0..1 range in respect to pose width
			landmark.y = (landmark.y - y_min) / (y_max-y_min)				# Normalize to 0..1 range in respect to pose height
			landmark.z = - 1 + 2 * (landmark.z - z_min) / (z_max-z_min)		# Normalize to -1..1 range in respect to pose depth

		return pose

	
if (__name__ == '__main__'):
	# mode and feature
	try:
		if sys.argv[1] in modes and sys.argv[2] in features:
			feature = sys.argv[2]

			if len(sys.argv) > 3 and sys.argv[3] in exercises:
				# stage dataset
				mode = 'stage_' + sys.argv[1]
				exercise = '/' + sys.argv[3]
				labels = stages
			else: 
				mode = sys.argv[1]
				exercise = ''	
				labels = exercises	
		else:
			raise Exception()
	except:
		print('Mode or feature not permitted. Please try again.')
		exit(0)

	source_dir = f'../../../BA/{mode}_data{exercise}'
	csv_path = f'../../../BA/{mode}_data{exercise}/{mode}_{feature}.csv'

	# recreate destination file
	if (os.path.exists(csv_path)):
		os.remove(csv_path)
	with open(csv_path, 'a', newline='') as file:
			print(f'Writing to {csv_path}')
			writer = csv.writer(file)
			if feature == 'angles':
				writer.writerow(['class'] + list(relevant_angles.keys()))
			else: 
				writer.writerow(['class'] + sum([[f'{x}_x', f'{x}_y', f'{x}_z'] for x in coco_body_parts.values()], []))

	inferencer = BlazePoseInferencer()
	for label in labels:
		print(f'Exercise set: {label}')
		for file_name in os.listdir(os.path.join(source_dir, label)):
			# skip video files
			if (os.path.isdir(os.path.join(source_dir, label, file_name))):
				print(f'Processing video {file_name}')
				for image_name in tqdm(os.listdir(os.path.join(source_dir, label, file_name))):
					# skip hidden files
					if image_name.startswith('.'):
						continue

					# run inference and write to file
					try: 
						image = cv2.imread(os.path.join(source_dir, label, file_name, image_name))
						image_height, image_width, _ = image.shape
						results = inferencer.run_inference_on(image, False)
						landmarks = results.pose_landmarks
						if (len(landmarks) > 0):
							if feature == 'angles':
								inferencer.calculate_and_write_angles(landmarks[0], label)
							elif feature == 'points':
								inferencer.write_pose(landmarks[0], label)
							else:
								inferencer.write_pose(landmarks[0], label, normalize=True)

					except Exception as e:
						print(e)
						print('Error occured at:')
						print(os.path.join(source_dir, label, file_name, image_name))
						exit(0)
		
	print(inferencer.total_inference_time/inferencer.total_inferences)
	print(inferencer.total_inferences)

	""" inferencer = BlazePoseInferencer()
	source_dir = '../../../BA/exercise_vids/selection'
	print(os.path.abspath(source_dir))
	for file_name in os.listdir(source_dir):
		if file_name.startswith('.'):
			continue
		print(file_name)
		image = cv2.imread(os.path.join(source_dir, file_name))
		results = inferencer.run_inference_on(image, False)
		#image = inferencer.draw_coco_body(image, results)
		landmark_image = inferencer.draw_landmarks_on_image(image, results)
		cv2.imwrite(os.path.join(source_dir, file_name.replace('.jpg', '_skeleton.jpg')), landmark_image) """
