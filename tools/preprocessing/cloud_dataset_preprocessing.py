import json
import math
import os
import random
import re
import shutil
from copy import deepcopy
from glob import glob

import cv2
import numpy as np
from plyfile import PlyData, PlyElement


# Spherical harmonics constant for RGB conversion
C0 = 0.28209479177387814


def storePly(path, xyz, rgb):
    """Store point cloud data in PLY format with positions and colors."""
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def SH2RGB(sh):
    """Convert spherical harmonics to RGB values."""
    return sh * C0 + 0.5


def split_dataset(img_files, test_ratio=0.1, seed=None):
    """Split dataset into train and test sets."""
    if seed is not None:
        random.seed(seed)

    random.shuffle(img_files)
    num_test_samples = int(len(img_files) * test_ratio)

    test_files = img_files[:num_test_samples]
    train_files = img_files[num_test_samples:]

    return train_files, test_files


def compute_camera_to_world_matrix(position, direction, up):
    """Compute camera-to-world transformation matrix from position and orientation vectors."""
    position = np.array(position)
    direction = np.array(direction)
    up = np.array(up)

    # Normalize direction and up vectors
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
    rotation_matrix[:3, 2] = direction

    # Create the translation matrix
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = position

    # Compute camera-to-world matrix
    camera_to_world_matrix = translation_matrix @ rotation_matrix
    camera_to_world_matrix[:3, 1:3] *= -1

    return camera_to_world_matrix


def read_camera_pose(file_path):
    """Read camera pose from file containing position, direction, and up vectors."""
    with open(file_path, 'r') as file:
        lines = file.readlines()
        position = list(map(float, lines[0].strip().split()))
        direction = list(map(float, lines[1].strip().split()))
        up = list(map(float, lines[2].strip().split()))

    return position, direction, up


def sort_numerically_xy(file_list):
    """Sort files into x and y groups, then sort each group numerically."""
    def extract_number(file_name):
        match = re.search(r'(\d+)', file_name)
        return int(match.group(1)) if match else float('inf')

    x_files = [f for f in file_list if '/x/' in f]
    y_files = [f for f in file_list if '/y/' in f]

    x_files_sorted = sorted(x_files, key=extract_number)
    y_files_sorted = sorted(y_files, key=extract_number)

    return x_files_sorted + y_files_sorted


def sort_numerically(file_list):
    """Sort files numerically based on the last number in the filename."""
    def extract_number(file_name):
        base_name = file_name.split('/')[-1]
        numbers = re.findall(r'\d+', base_name)
        return int(numbers[-1]) if numbers else float('inf')

    return sorted(file_list, key=extract_number)


if __name__ == "__main__":
    root = "/code/dataset/scatter/subsurface_suzanne"

    # Create output directories
    os.makedirs(os.path.join(root, "train"), exist_ok=True)
    os.makedirs(os.path.join(root, "test"), exist_ok=True)
    os.makedirs(os.path.join(root, "val"), exist_ok=True)

    file_path_train = os.path.join(root, 'transforms_train.json')
    file_path_test = os.path.join(root, 'transforms_test.json')
    file_path_val = os.path.join(root, 'transforms_val.json')
    file_path = os.path.join(root, 'transforms.json')

    # Point cloud bounds for different datasets:
    # cloud: [-2.7, 2.7]
    # explosion: [-1.4, 1.4]
    # suzanne: [-1.4, 1.4] + [0, 0, 0.54]
    # bunny_cloud: [-2.4, 2.4]

    # Generate random point cloud for suzanne
    num_pts = 100000
    xyz = np.random.random((num_pts, 3)) * 2.8 - 1.4 + np.array([0, 0, 0.54])[None]

    shs = np.random.random((xyz.shape[0], 3)) / 255.0
    storePly(os.path.join(root, "points3d.ply"), xyz, SH2RGB(shs) * 255)

    # Load and split images
    img_files = glob(os.path.join(root, "images/*.png"))
    num_test_samples = 50
    train_files = img_files[num_test_samples:]
    test_files = img_files[:num_test_samples]

    # Load base JSON data
    with open(file_path, 'r') as file:
        json_data = json.load(file)

    # Create separate JSON data for train, test, and validation
    json_data_train = deepcopy(json_data)
    json_data_test = deepcopy(json_data)
    json_data_train['frames'] = json_data_train['frames'][num_test_samples:]
    json_data_test['frames'] = json_data_test['frames'][:num_test_samples]
    json_data_val = deepcopy(json_data_test)

    # Update file paths in JSON data
    for frame in json_data_train['frames']:
        frame['file_path'] = frame['file_path'][:-4]
    for frame in json_data_test['frames']:
        frame['file_path'] = frame['file_path'].replace('train', 'test')[:-4]
    for frame in json_data_val['frames']:
        frame['file_path'] = frame['file_path'].replace('train', 'val')[:-4]

    # Copy training images
    for train_file in train_files:
        name = os.path.join(root, "train", train_file.split('/')[-1])
        shutil.copy(train_file, name)

    # Copy test images to both test and val directories
    for test_file in test_files:
        test_name = os.path.join(root, "test", test_file.split('/')[-1])
        val_name = os.path.join(root, "val", test_file.split('/')[-1])
        shutil.copy(test_file, test_name)
        shutil.copy(test_file, val_name)

    # Save JSON files
    with open(file_path_train, 'w') as file:
        json.dump(json_data_train, file, indent=4)
    with open(file_path_test, 'w') as file:
        json.dump(json_data_test, file, indent=4)
    with open(file_path_val, 'w') as file:
        json.dump(json_data_val, file, indent=4)

    print(f"Processed {len(img_files)} images")
