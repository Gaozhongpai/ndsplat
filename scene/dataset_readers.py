#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from utils.graphics_utils import BasicPointCloud
from tqdm import tqdm

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    x_threshold: float = None
    label: list = None
    color_idx: float = None
    timestamp: float = 0.0  # Time dimension for 7DGS

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for key in tqdm(cam_extrinsics, desc="Reading camera metadata"):
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE" or "SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]

        cam_infos.append(
            CameraInfo(
                uid=uid,
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                image_path=image_path,
                image_name=image_name,
                width=width,
                height=height,
                x_threshold=None,
                label=None,
                color_idx=None,
                timestamp=0.0,
            )
        )
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

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

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        if "train" in transformsfile:
            frames = contents["frames"] # [:2]
        else:
            frames = contents["frames"] # [:20]

        for idx, frame in enumerate(tqdm(frames, desc="Reading frame metadata")):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # Read cutting plane parameters if they exist in the JSON
            x_threshold = frame.get("x_threshold", None)
            color_idx = frame.get("color_idx", None)
            label = frame.get("label", None)

            # Read time parameter for 7DGS if it exists
            try:
                timestamp = frame["time"]
            except:
                timestamp = 0.0

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            with Image.open(image_path) as image:
                width, height = image.size

                # # Check if the image is totally black
                # # Convert to numpy array and check if all pixels are black
                # img_array = np.array(image)
                # if img_array.max() == 0:
                #     print(f"Skipping totally black frame: {image_name} (timestamp: {timestamp})")
                #     continue

            fovy = focal2fov(fov2focal(fovx, width), height)
            cam_infos.append(
                CameraInfo(
                    uid=idx,
                    R=R,
                    T=T,
                    FovY=fovy,
                    FovX=fovx,
                    image_path=image_path,
                    image_name=image_name,
                    width=width,
                    height=height,
                    x_threshold=x_threshold,
                    color_idx=color_idx,
                    label=label,
                    timestamp=timestamp,
                )
            )

    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    # For D-NeRF lego scene, use transforms_val.json instead of transforms_test.json (same as 4DGS)
    is_dnerf_lego = 'dnerf' in path.lower() and 'lego' in path.lower()
    test_file = "transforms_val.json" if is_dnerf_lego else "transforms_test.json"
    test_cam_infos = readCamerasFromTransforms(path, test_file, white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readDF3DVCameras(cam_extrinsics, cam_intrinsics, images_folder):
    """Read DF3DV cameras, rescaling intrinsics to match the on-disk (downsampled)
    image resolution. COLMAP poses are stored at full resolution, but
    undistortion_images_8/ holds images downsampled by ~8x, so FoV must be
    derived from the actual image dimensions rather than the COLMAP width/height.
    """
    cam_infos = []
    for key in tqdm(cam_extrinsics, desc="Reading DF3DV camera metadata"):
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]

        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        if not os.path.exists(image_path):
            # Some scenes keep .JPG in COLMAP but downsampled files share the
            # same basename/extension; skip cleanly if the image is absent.
            continue
        image_name = os.path.basename(image_path).split(".")[0]

        # Actual (downsampled) resolution on disk.
        with Image.open(image_path) as img:
            width, height = img.size

        # FoV is resolution-independent: compute from the COLMAP focal length
        # scaled to the on-disk resolution.
        colmap_w, colmap_h = intr.width, intr.height
        if intr.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
            fx = intr.params[0]
            fy = intr.params[0]
        elif intr.model == "PINHOLE":
            fx = intr.params[0]
            fy = intr.params[1]
        else:
            assert False, f"DF3DV: unsupported COLMAP camera model {intr.model} (expect undistorted PINHOLE/SIMPLE_PINHOLE)"

        sx = width / colmap_w
        sy = height / colmap_h
        FovX = focal2fov(fx * sx, width)
        FovY = focal2fov(fy * sy, height)

        cam_infos.append(
            CameraInfo(
                uid=intr.id,
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                image_path=image_path,
                image_name=image_name,
                width=width,
                height=height,
                x_threshold=None,
                label=None,
                color_idx=None,
                timestamp=0.0,
            )
        )
    return cam_infos


def readDF3DVSceneInfo(path, images, eval, llffhold=8):
    """Load a DF3DV-41 / DF3DV-1K scene.

    Layout (per scene, inside the <scene>-All directory passed as `path`):
        undistortion_sparse/0/{cameras,images,points3D}.{bin,txt}
        undistortion_images_8/{clutter_*,extra_*}.JPG   (downsampled by 8)
        split.json   (optional; clutter=train / extra=eval)

    Train = clutter_* images, Eval = extra_* (clean) images. The split is keyed
    on the image-name prefix, which the benchmark's leaderboard tooling relies on.
    """
    sparse_dir = os.path.join(path, "undistortion_sparse", "0")
    try:
        cam_extrinsics = read_extrinsics_binary(os.path.join(sparse_dir, "images.bin"))
        cam_intrinsics = read_intrinsics_binary(os.path.join(sparse_dir, "cameras.bin"))
    except Exception:
        cam_extrinsics = read_extrinsics_text(os.path.join(sparse_dir, "images.txt"))
        cam_intrinsics = read_intrinsics_text(os.path.join(sparse_dir, "cameras.txt"))

    reading_dir = images if images not in (None, "images") else "undistortion_images_8"
    cam_infos_unsorted = readDF3DVCameras(
        cam_extrinsics=cam_extrinsics,
        cam_intrinsics=cam_intrinsics,
        images_folder=os.path.join(path, reading_dir),
    )
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    # Optional split.json cross-check (list of clean/clutter image names).
    split_clean = None
    split_json = os.path.join(path, "split.json")
    if os.path.exists(split_json):
        try:
            with open(split_json) as f:
                sj = json.load(f)
            # Be liberal about key naming; collect any list that names clean/extra images.
            clean = []
            for k, v in sj.items() if isinstance(sj, dict) else []:
                if isinstance(v, list) and ("clean" in k.lower() or "extra" in k.lower() or "test" in k.lower() or "eval" in k.lower()):
                    clean.extend(os.path.basename(str(x)).split(".")[0] for x in v)
            if clean:
                split_clean = set(clean)
        except Exception as e:
            print(f"DF3DV: could not parse split.json ({e}); falling back to filename prefixes.")

    def is_eval(cam):
        name = cam.image_name.lower()
        if split_clean is not None:
            return cam.image_name in split_clean
        return name.startswith("extra")

    def is_train(cam):
        name = cam.image_name.lower()
        return name.startswith("clutter") or (not is_eval(cam))

    if eval:
        train_cam_infos = [c for c in cam_infos if c.image_name.lower().startswith("clutter")]
        test_cam_infos = [c for c in cam_infos if is_eval(c)]
        # Fallback for scenes that don't use the clutter_ prefix.
        if not train_cam_infos:
            train_cam_infos = [c for c in cam_infos if not is_eval(c)]
    else:
        # No eval: train on everything available (clutter + extra).
        train_cam_infos = cam_infos
        test_cam_infos = []

    print(f"DF3DV scene: {len(train_cam_infos)} train (clutter), {len(test_cam_infos)} eval (extra) cameras")

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(sparse_dir, "points3D.ply")
    bin_path = os.path.join(sparse_dir, "points3D.bin")
    txt_path = os.path.join(sparse_dir, "points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting points3D.bin to .ply (first load only)...")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except Exception:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except Exception:
        pcd = None

    return SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "DF3DV": readDF3DVSceneInfo,
}
