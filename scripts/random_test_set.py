import os
import random
import shutil
from pathlib import Path
from tqdm import tqdm

exercise_types = ['pull-up', 'push-up', 'squat']
source_dir = '../../../BA/training_data'
dest_dir = '../../../BA/training_data/random_set'
max_images = 500

Path(dest_dir).mkdir(parents=True, exist_ok=True)
for count in tqdm(range(max_images)):
    exercise = random.choice(exercise_types)
    dir_list = os.listdir(os.path.join(source_dir, exercise))
    while True:
        vid_name = random.choice(dir_list)
        if (os.path.isdir(os.path.join(source_dir, exercise, vid_name))):
            break
    
    file_list = os.listdir(os.path.join(source_dir, exercise, vid_name))

    while True: 
        file_name = random.choice(file_list)
        if (f'{vid_name}_{file_name}' not in os.listdir(dest_dir)):
            break

    shutil.copyfile(os.path.join(source_dir, exercise, vid_name, file_name), os.path.join(dest_dir, f'{vid_name}_{file_name}'))


