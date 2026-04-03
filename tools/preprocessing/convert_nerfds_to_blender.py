"""Convert NeRF-DS (HyperNeRF/Nerfies format) to NeRF Synthetic (Blender) format.

NeRF-DS format:
  - camera/{id}.json: per-frame camera (orientation, position, focal_length, etc.)
  - rgb/{scale}x/{id}.png: images at various scales
  - metadata.json: per-frame metadata (time_id, warp_id, etc.)
  - dataset.json: train/val split
  - scene.json: scene center, scale, near, far

Output (Blender format):
  - transforms_train.json / transforms_test.json with camera_angle_x and frames
  - images symlinked or copied
"""

import argparse
import json
import math
import os
import shutil
import struct

import numpy as np


def load_camera_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def nerfds_camera_to_c2w(cam_json, scene_center, scene_scale):
    """Convert NeRF-DS camera (orientation, position) to 4x4 camera-to-world matrix.

    NeRF-DS convention:
    - orientation: 3x3 rotation matrix (world-to-camera rotation)
    - position: camera position in world coordinates

    Output: 4x4 camera-to-world in OpenGL/Blender convention (Y up, Z back)
    """
    orientation = np.array(cam_json['orientation'])  # [3, 3] world-to-camera rotation
    position = np.array(cam_json['position'])  # [3] world position

    # Apply scene normalization (center and scale)
    position = (position - scene_center) * scene_scale

    # World-to-camera: R @ (x - t) = R @ x - R @ t
    # So w2c rotation = orientation, w2c translation = -orientation @ position
    R_w2c = orientation
    t_w2c = -R_w2c @ position

    # Build 4x4 world-to-camera
    w2c = np.eye(4)
    w2c[:3, :3] = R_w2c
    w2c[:3, 3] = t_w2c

    # Camera-to-world
    c2w = np.linalg.inv(w2c)

    # Convert from COLMAP/OpenCV convention (Y down, Z forward)
    # to OpenGL/Blender convention (Y up, Z back)
    c2w[:3, 1:3] *= -1

    return c2w


def convert_scene(scene_dir, output_dir, image_scale=2):
    """Convert a single NeRF-DS scene to Blender format."""
    # Load scene info
    with open(os.path.join(scene_dir, 'scene.json'), 'r') as f:
        scene_info = json.load(f)
    scene_center = np.array(scene_info['center'])
    scene_scale = scene_info['scale']

    # Load dataset split
    with open(os.path.join(scene_dir, 'dataset.json'), 'r') as f:
        dataset = json.load(f)
    train_ids = [str(i) for i in dataset['train_ids']]
    val_ids = [str(i) for i in dataset['val_ids']]

    # Load metadata for timestamps
    metadata = {}
    metadata_path = os.path.join(scene_dir, 'metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

    # Get max time_id for normalization
    all_ids = train_ids + val_ids
    max_time_id = 0
    for item_id in all_ids:
        if item_id in metadata and 'time_id' in metadata[item_id]:
            max_time_id = max(max_time_id, metadata[item_id]['time_id'])
    if max_time_id == 0:
        max_time_id = 1  # avoid division by zero

    # Get camera_angle_x from first camera
    first_cam_path = os.path.join(scene_dir, 'camera', f'{train_ids[0]}.json')
    first_cam = load_camera_json(first_cam_path)
    focal_length = first_cam['focal_length']
    image_width = first_cam['image_size'][0]
    # Scale focal length by image scale
    focal_length_scaled = focal_length / image_scale
    image_width_scaled = image_width // image_scale
    camera_angle_x = 2.0 * math.atan(image_width_scaled / (2.0 * focal_length_scaled))

    os.makedirs(output_dir, exist_ok=True)

    # Process train and test splits
    for split_name, item_ids in [('train', train_ids), ('test', val_ids)]:
        frames = []
        img_dir = os.path.join(output_dir, split_name)
        os.makedirs(img_dir, exist_ok=True)

        for idx, item_id in enumerate(item_ids):
            # Load camera
            cam_path = os.path.join(scene_dir, 'camera', f'{item_id}.json')
            if not os.path.exists(cam_path):
                continue
            cam_json = load_camera_json(cam_path)

            # Convert to c2w matrix
            c2w = nerfds_camera_to_c2w(cam_json, scene_center, scene_scale)

            # Get timestamp (normalized to [0, 1])
            timestamp = 0.0
            if item_id in metadata and 'time_id' in metadata[item_id]:
                timestamp = metadata[item_id]['time_id'] / max_time_id

            # Copy/symlink image
            src_img = os.path.join(scene_dir, 'rgb', f'{image_scale}x', f'{item_id}.png')
            if not os.path.exists(src_img):
                continue
            dst_img = os.path.join(img_dir, f'{item_id}.png')
            if not os.path.exists(dst_img):
                os.symlink(os.path.abspath(src_img), dst_img)

            frame = {
                'file_path': f'./{split_name}/{item_id}',
                'transform_matrix': c2w.tolist(),
                'time': timestamp,
            }
            frames.append(frame)

        # Write transforms JSON
        transforms = {
            'camera_angle_x': camera_angle_x,
            'frames': frames,
        }
        output_path = os.path.join(output_dir, f'transforms_{split_name}.json')
        with open(output_path, 'w') as f:
            json.dump(transforms, f, indent=2)

        print(f'  {split_name}: {len(frames)} frames -> {output_path}')

    # Convert point cloud if available
    points_npy = os.path.join(scene_dir, 'points.npy')
    ply_path = os.path.join(output_dir, 'points3d.ply')
    if os.path.exists(points_npy) and not os.path.exists(ply_path):
        pts = np.load(points_npy)
        # Apply scene normalization (same as cameras)
        pts = (pts - scene_center) * scene_scale
        # Random colors
        colors = (np.random.random((len(pts), 3)) * 255).astype(np.uint8)
        # Write PLY
        _write_ply(ply_path, pts, colors)
        print(f'  points: {len(pts)} points -> {ply_path}')


def _write_ply(path, xyz, rgb):
    """Write a PLY point cloud file with normals (required by fetchPly)."""
    n = len(xyz)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float nx\n"
        "property float ny\n"
        "property float nz\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        for i in range(n):
            f.write(struct.pack('<fff', xyz[i, 0], xyz[i, 1], xyz[i, 2]))
            f.write(struct.pack('<fff', 0.0, 0.0, 0.0))  # zero normals
            f.write(struct.pack('<BBB', rgb[i, 0], rgb[i, 1], rgb[i, 2]))


def main():
    parser = argparse.ArgumentParser(description='Convert NeRF-DS to Blender format')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Root directory containing NeRF-DS scenes')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for converted scenes')
    parser.add_argument('--image_scale', type=int, default=2,
                        help='Image scale factor (default: 2)')
    args = parser.parse_args()

    # Process each scene
    for scene_name in sorted(os.listdir(args.input_dir)):
        scene_dir = os.path.join(args.input_dir, scene_name)
        if not os.path.isdir(scene_dir):
            continue
        dataset_json = os.path.join(scene_dir, 'dataset.json')
        if not os.path.exists(dataset_json):
            continue

        print(f'Converting {scene_name}...')
        output_scene_dir = os.path.join(args.output_dir, scene_name)
        convert_scene(scene_dir, output_scene_dir, args.image_scale)


if __name__ == '__main__':
    main()
