import numpy as np
from glob import glob
import os 
import random
import json
import shutil
import re
import math
import SimpleITK as sitk
from skimage import measure
from plyfile import PlyData, PlyElement
from copy import deepcopy
import cv2

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

C0 = 0.28209479177387814
def SH2RGB(sh):
    return sh * C0 + 0.5

def split_dataset(img_files, test_ratio=0.1, seed=None):
    # Set the random seed for repeatability
    if seed is not None:
        random.seed(seed)
    
    # Shuffle the list of images
    random.shuffle(img_files)
    
    # Compute the number of test samples
    num_test_samples = int(len(img_files) * test_ratio)
    
    # Split the dataset
    test_files = img_files[:num_test_samples]
    train_files = img_files[num_test_samples:]
    
    return train_files, test_files

import numpy as np

def compute_camera_to_world_matrix(position, direction, up):
    position = np.array(position)
    direction = np.array(direction)
    up = np.array(up)
    
    # Ensure direction and up vectors are normalized
    direction = direction / np.linalg.norm(direction)
    up = up / np.linalg.norm(up)
    
    # Compute the right vector as the cross product of direction and up
    right = np.cross(direction, up)
    right = right / np.linalg.norm(right)
    
    # Recompute the up vector to ensure orthogonality
    up = np.cross(direction, right)
    
    # Create the rotation matrix for OpenGL/Blender (Y up, Z back)
    rotation_matrix = np.eye(4)
    rotation_matrix[:3, 0] = right
    rotation_matrix[:3, 1] = up
    rotation_matrix[:3, 2] = direction  # Positive Z is back in OpenGL/Blender
    
    # Create the translation matrix
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = position
    
    # Compute camera-to-world matrix
    camera_to_world_matrix = translation_matrix @ rotation_matrix
    
    camera_to_world_matrix[:3, 1:3] *= -1
    
    return camera_to_world_matrix

def read_camera_pose(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
        # Parse the lines into position, direction, and up vectors
        position = list(map(float, lines[0].strip().split()))
        direction = list(map(float, lines[1].strip().split()))
        up = list(map(float, lines[2].strip().split()))
        
    return position, direction, up

def sort_numerically_xy(file_list):
    def extract_number(file_name):
        match = re.search(r'(\d+)', file_name)
        return int(match.group(1)) if match else float('inf')
    
    # Separate files into two groups: x_ and y_
    x_files = [f for f in file_list if '/x/' in f]
    y_files = [f for f in file_list if '/y/' in f]
    
    # Sort each group numerically
    x_files_sorted = sorted(x_files, key=extract_number)
    y_files_sorted = sorted(y_files, key=extract_number)
    
    # Combine the two sorted groups
    return x_files_sorted + y_files_sorted

def sort_numerically(file_list):
    def extract_number(file_name):
        # Extract the base filename without the path
        base_name = file_name.split('/')[-1]
        # Find all numbers in the base filename
        numbers = re.findall(r'\d+', base_name)
        # Return the last number found, or float('inf') if no numbers
        return int(numbers[-1]) if numbers else float('inf')
    
    # Sort each group numerically
    file_list_sorted = sorted(file_list, key=extract_number)
    
    # Combine the two sorted groups
    return file_list_sorted 

    
import SimpleITK as sitk
import numpy as np
from skimage import measure

if __name__ == "__main__":
    root = "/code/dataset/scatter/subsurface_suzanne"

    os.makedirs(os.path.join(root, "train"), exist_ok=True)
    os.makedirs(os.path.join(root, "test"), exist_ok=True)
    os.makedirs(os.path.join(root, "val"), exist_ok=True)
    file_path_train = os.path.join(root, 'transforms_train.json')
    file_path_test = os.path.join(root, 'transforms_test.json')
    file_path_val = os.path.join(root, 'transforms_val.json')
    file_path = os.path.join(root, 'transforms.json')
    
    ## cloud [-2.7, 2.7]
    ## explosion [-1.4, 1.4]                     # old [-2.8, 2.8]
    ## suzanne  [-1.4, 1.4] z+ 0.54              # old [-2.7, 2.7] z +1
    ## bunny_cloud [-2.4, 2.4]
    
    num_pts = 100000
    # Generate point cloud
    # xyz = np.random.random((num_pts, 3)) * 5.4 - 2.7 ## cloud and smoke
    # xyz = np.random.random((num_pts, 3)) * 2.8 - 1.4 ## explosion
    xyz = np.random.random((num_pts, 3)) * 2.8 - 1.4 + np.array([0, 0, 0.54])[None] ## suzanne 
    # xyz = np.random.random((num_pts, 3)) * 3.2 - 1.6 + np.array([-0.11, 0.09, 1.09])[None] ## dragon 
    # xyz = np.random.random((num_pts, 3)) * 4.8 - 2.4 ## bunny_cloud
    
    shs = np.random.random((xyz.shape[0], 3)) / 255.0
    storePly(os.path.join(root, "points3d.ply"), xyz, SH2RGB(shs) * 255)

    img_files = glob(os.path.join(root, "images/*.png"))
    # img_files = img_files + glob(os.path.join(root, "y/*.png"))
    
    # Compute the number of test samples
    num_test_samples = 50
    train_files, test_files = img_files[num_test_samples:], img_files[:num_test_samples]    
    with open(file_path, 'r') as file:
        json_data = json.load(file)
    
    # nd = 8
    # json_data["camera_angle_x"] = math.tanh(math.tan(json_data["camera_angle_x"]) * (1 - 2 / nd) )
    
    camera_angle_x_rad = math.radians(60) # 2 * np.arctan(0.5 * W / focal_len)
    json_data_train = deepcopy(json_data)
    json_data_test = deepcopy(json_data)
    json_data_train['frames'] = json_data_train['frames'][num_test_samples:]
    json_data_test['frames'] = json_data_test['frames'][:num_test_samples]
    json_data_val = deepcopy(json_data_test)
    
    for frame in json_data_train['frames']:
        frame['file_path'] = frame['file_path'][:-4]
    for frame in json_data_test['frames']:
        frame['file_path'] = frame['file_path'].replace('train', 'test')[:-4]
    for frame in json_data_val['frames']:
        frame['file_path'] = frame['file_path'].replace('train', 'val')[:-4]

    for idx, train_file in enumerate(train_files):
        im = cv2.imread(train_file)
        H, W, _ = im.shape
        # pad = int(H/nd)
        # im = im[pad:-pad, pad:-pad]
        name = os.path.join(root, "train", train_file.split('/')[-1])
        # cv2.imwrite(name, im)
        shutil.copy(train_file, name)
        
    for idx, test_file in enumerate(test_files):
        im = cv2.imread(test_file)
        H, W, _ = im.shape
        # pad = int(H/nd)
        # im = im[pad:-pad, pad:-pad]
        name = os.path.join(root, "test", test_file.split('/')[-1])
        # cv2.imwrite(name, im)
        shutil.copy(test_file, name)
    
        name = os.path.join(root, "val", test_file.split('/')[-1])
        # cv2.imwrite(name, im)
        shutil.copy(test_file, name)
    
    with open(file_path_train, 'w') as file:
        json.dump(json_data_train, file, indent=4)
    with open(file_path_test, 'w') as file:
        json.dump(json_data_test, file, indent=4)
    with open(file_path_val, 'w') as file:
        json.dump(json_data_val, file, indent=4)

    print(len(img_files))