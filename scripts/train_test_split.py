import os
import pathlib
import random
import shutil
from tqdm import tqdm

splits = {'train': 0.7, 'valid': 0.75, 'test': 1}
exercises = ['plank', 'pull Up', 'push-up', 'squat']
source_dir = '../../../BA/BA_2023/action_classifier_dataset'
target_dir = '../../../BA/BA_2023/action_classifier_dataset'

def train_test_split():
    # clear out existing subdirs
    shutil.rmtree(target_dir)
    os.makedirs(target_dir)

    for exercise in exercises:
        all_files = os.listdir(source_dir + '/' + exercise)
        for split, percentage in splits.items():
            print(f'Copying {exercise} {split} files')
            # create folder if not there yet
            pathlib.Path(os.path.join(target_dir, split, exercise)).mkdir(parents=True, exist_ok=True)
            file_count = percentage * len(all_files)

            # copy randomly selected files to destination
            for count in tqdm(range(0, int(file_count))):
                index = random.randint(0, len(all_files)-1)
                file_name = all_files[index]
                shutil.copy(os.path.join(source_dir, exercise, file_name), os.path.join(target_dir, split, exercise, file_name))
                all_files.pop(index)

if (__name__ == '__main__'):
    train_test_split()