import os
import shutil
import cv2


#source_dir = '../../../BA/BA_2023/action_classifier_dataset/train/plank'
source_dir = '../../../BA/comparison_vids'

def videoToFrames():
    for file_name in os.listdir(source_dir):
        if (file_name == '.DS_Store'): 
            continue
        print(f'Read a new file: {file_name}')
        target_dir = source_dir + '/' + file_name.split('.')[0]
        # clear out existing subdirs
        if(os.path.exists(target_dir)):
            shutil.rmtree(target_dir)
        os.makedirs(target_dir)

        vid = cv2.VideoCapture(source_dir + '/' + file_name)
        success, image = vid.read()
        count = 0
        while success:
            cv2.imwrite(f'{target_dir}/frame{count}.jpg', image)
            success, image = vid.read()
            print('Read a new frame: ', success)
            count += 1

if (__name__ == '__main__'):
    videoToFrames()