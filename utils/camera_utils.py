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
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

from scene.cameras import Camera
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.io import read_image, encode_jpeg, decode_jpeg
from tqdm import tqdm
from utils.graphics_utils import fov2focal

WARNED = False

def loadCam(args, id, cam_info, resolution_scale, *, resolution=None, image_tensor=None, alpha_mask=None, compressed_data=None):
    if resolution is None:
        resolution = _compute_target_resolution(args, cam_info, resolution_scale)

    if image_tensor is None and compressed_data is None:
        image_tensor, alpha_mask = _load_image_tensor(
            cam_info.image_path,
            resolution,
            args.white_background
        )

    gt_image = image_tensor[:3, ...] if image_tensor is not None else None
    loaded_mask = alpha_mask

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id,
                  x_threshold=cam_info.x_threshold, color_idx=cam_info.color_idx, label=cam_info.label,
                  data_device=args.data_device, timestamp=cam_info.timestamp,
                  compressed_data=compressed_data)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = [None] * len(cam_infos)
    if not cam_infos:
        return camera_list

    # Determine whether to use compression (default: auto-enable for datasets > 1000 images)
    use_jpeg_compression = getattr(args, 'use_jpeg_compression', len(cam_infos) > 1000)

    # Safety override: force enable for very large datasets to prevent OOM
    if not use_jpeg_compression and len(cam_infos) > 1000:
        print(f"[WARNING] Compression disabled but dataset has {len(cam_infos)} images (>1000).")
        print(f"[WARNING] Force-enabling compression to prevent GPU OOM.")
        use_jpeg_compression = True

    if use_jpeg_compression:
        print(f"[INFO] Using JPEG compression for {len(cam_infos)} images to reduce GPU memory usage (~8-10x savings)")
    else:
        print(f"[INFO] Loading {len(cam_infos)} images directly to GPU (use --use_jpeg_compression to save memory)")

    max_workers = 4 # min(32, (os.cpu_count() or 1) * 2)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_prepare_camera_payload, args, idx, cam_info, resolution_scale, use_jpeg_compression): idx
            for idx, cam_info in enumerate(cam_infos)
        }
        with tqdm(total=len(futures), desc="Preparing cameras") as progress:
            for future in as_completed(futures):
                idx = futures[future]
                if use_jpeg_compression:
                    resolution, compressed_data, alpha_mask = future.result()
                    camera_list[idx] = loadCam(
                        args,
                        idx,
                        cam_infos[idx],
                        resolution_scale,
                        resolution=resolution,
                        compressed_data=compressed_data,
                        alpha_mask=alpha_mask,
                    )
                else:
                    resolution, image_tensor, alpha_mask = future.result()
                    camera_list[idx] = loadCam(
                        args,
                        idx,
                        cam_infos[idx],
                        resolution_scale,
                        resolution=resolution,
                        image_tensor=image_tensor,
                        alpha_mask=alpha_mask,
                    )
                progress.update(1)

    return camera_list


def _load_image_tensor(image_path, resolution, white_background):
    """
    Load an image from disk, optionally composite alpha, and resize using torch-native ops.
    """
    image = read_image(image_path).float() / 255.0  # [C, H, W]

    alpha_mask = None
    if image.shape[0] == 4:
        alpha = image[3:4]
        rgb = image[:3]
        if white_background:
            rgb = rgb * alpha + (1.0 - alpha)
            alpha_mask = None
        else:
            rgb = rgb * alpha
            alpha_mask = alpha
    else:
        rgb = image

    if (resolution[0], resolution[1]) != (rgb.shape[2], rgb.shape[1]):
        size_hw = (resolution[1], resolution[0])
        rgb = F.interpolate(rgb.unsqueeze(0), size=size_hw, mode="bilinear", align_corners=False).squeeze(0)
        if alpha_mask is not None:
            alpha_mask = F.interpolate(alpha_mask.unsqueeze(0), size=size_hw, mode="bilinear", align_corners=False).squeeze(0)

    rgb = rgb.clamp(0.0, 1.0)
    return rgb, alpha_mask


def _compute_target_resolution(args, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.width, cam_info.height

    if args.resolution in [1, 2, 4, 8]:
        return (
            round(orig_w / (resolution_scale * args.resolution)),
            round(orig_h / (resolution_scale * args.resolution)),
        )

    if args.resolution == -1:
        if orig_w > 1600:
            global WARNED
            if not WARNED:
                print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                      "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                WARNED = True
            global_down = orig_w / 1600
        else:
            global_down = 1
    else:
        global_down = orig_w / args.resolution

    scale = float(global_down) * float(resolution_scale)
    return (int(orig_w / scale), int(orig_h / scale))


def _prepare_camera_payload(args, idx, cam_info, resolution_scale, use_jpeg_compression=False):
    resolution = _compute_target_resolution(args, cam_info, resolution_scale)

    if use_jpeg_compression:
        # Optimization: Check if we can use original bytes without decoding
        file_ext = os.path.splitext(cam_info.image_path)[1].lower()
        orig_size = (cam_info.width, cam_info.height)

        if file_ext in ['.jpg', '.jpeg'] and orig_size == resolution:
            # FAST PATH (JPEG): Use original JPEG bytes directly - NO DECODING!
            # This avoids wasting time/memory decoding a JPEG just to throw away the tensor
            with open(cam_info.image_path, 'rb') as f:
                jpeg_bytes = f.read()

            # Determine shape (assume RGB, alpha handling happens at decompression)
            shape = (3, resolution[1], resolution[0])

            compressed_data = {
                'format': 'jpeg_original',
                'bytes': jpeg_bytes,
                'has_alpha': False,
                'shape': shape
            }
            alpha_mask = None
            return resolution, compressed_data, alpha_mask

        elif file_ext == '.png' and orig_size == resolution and not args.white_background:
            # FAST PATH (PNG): Use original PNG bytes directly - NO DECODING!
            # Only works if no alpha compositing needed (white_background=False)
            with open(cam_info.image_path, 'rb') as f:
                png_bytes = f.read()

            # Check if PNG has alpha channel
            from PIL import Image
            with Image.open(cam_info.image_path) as img:
                has_alpha = img.mode in ('RGBA', 'LA', 'PA')
                # Determine actual channels
                if img.mode in ('RGBA', 'LA'):
                    channels = 4
                elif img.mode in ('RGB', 'L'):
                    channels = 3
                else:
                    channels = 3  # Default to RGB

            shape = (channels, resolution[1], resolution[0])

            compressed_data = {
                'format': 'png_original',
                'bytes': png_bytes,
                'has_alpha': has_alpha,
                'shape': shape
            }
            alpha_mask = None
            return resolution, compressed_data, alpha_mask

        else:
            # SLOW PATH: Need to decode, resize, or re-encode
            image_tensor, alpha_mask = _load_image_tensor(cam_info.image_path, resolution, args.white_background)
            compressed_data = _compress_image_smart(image_tensor, alpha_mask)
            return resolution, compressed_data, alpha_mask
    else:
        # No compression - load full tensor
        image_tensor, alpha_mask = _load_image_tensor(cam_info.image_path, resolution, args.white_background)
        return resolution, image_tensor, alpha_mask


def _compress_image_smart(image_tensor, alpha_mask):
    """
    Compress processed images (resized, alpha-composited, etc.) using PNG.

    Strategy: Use PNG compression for processed images since:
    - PNG is lossless (no additional artifacts after processing)
    - Preserves processed result perfectly
    - Avoids compounding quality loss from multiple JPEG re-encodings

    Trade-off: PNG may be 2-5x larger than JPEG, but guarantees zero additional
    quality degradation. For training 3D models where image quality matters, this
    is an acceptable trade-off.

    NOTE: This function is only called for processed images (resized/composited).
    For JPEG files at original resolution, the fast path in _prepare_camera_payload
    should be used instead to avoid redundant decoding.

    Args:
        image_tensor: [3, H, W] float tensor in [0, 1] (already processed)
        alpha_mask: [1, H, W] float tensor or None

    Returns:
        dict with format info and compressed data
    """
    # For all processed images: use PNG (lossless)
    # Once an image has been decoded and processed (resized/composited),
    # using PNG avoids adding NEW compression artifacts on top of processing artifacts
    from PIL import Image
    import io

    # Convert tensor to uint8
    rgb_uint8 = (image_tensor * 255.0).clamp(0, 255).byte().cpu().numpy()
    rgb_uint8 = np.transpose(rgb_uint8, (1, 2, 0))  # [H, W, C]

    # Encode as PNG (lossless)
    img = Image.fromarray(rgb_uint8, mode='RGB')
    buffer = io.BytesIO()
    img.save(buffer, format='PNG', compress_level=6)  # Good compression/speed tradeoff
    png_bytes = buffer.getvalue()

    return {
        'format': 'png_processed',
        'bytes': png_bytes,
        'has_alpha': alpha_mask is not None,
        'shape': image_tensor.shape
    }


def _decompress_jpeg_to_tensor(compressed_data):
    """
    Decompress image bytes back to float tensor.

    Supported formats:
    - 'jpeg_original': Original JPEG at full resolution (fast path, zero artifacts)
    - 'png_original': Original PNG at full resolution (fast path, lossless)
    - 'png_processed': Processed images (resized/composited) stored as PNG (lossless)

    Args:
        compressed_data: dict with 'bytes' and 'format' keys

    Returns:
        image_tensor: [3, H, W] or [4, H, W] float tensor in [0, 1]
    """
    format_type = compressed_data.get('format')
    if format_type is None:
        raise ValueError("Compressed data must contain 'format' key")

    # Get the bytes data
    image_bytes = compressed_data.get('bytes')
    if image_bytes is None:
        raise ValueError("Compressed data must contain 'bytes' key")

    # Handle different formats
    if format_type == 'jpeg_original':
        # Decode JPEG bytes using PyTorch (fast path - original JPEG bytes)
        jpeg_bytes = torch.frombuffer(image_bytes, dtype=torch.uint8)
        rgb_uint8 = decode_jpeg(jpeg_bytes)
        image_tensor = rgb_uint8.float() / 255.0

    elif format_type in ['png_original', 'png_processed']:
        # Decode PNG bytes using PIL
        # 'png_original' = original PNG at full resolution (fast path, zero decode/re-encode)
        # 'png_processed' = processed images (resized/composited) stored losslessly as PNG
        from PIL import Image
        import io

        buffer = io.BytesIO(image_bytes)
        img = Image.open(buffer)
        img_array = np.array(img)  # [H, W, C] or [H, W, C] with alpha

        # Convert to tensor [C, H, W]
        image_tensor = torch.from_numpy(img_array).float() / 255.0
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
        else:
            # Grayscale case
            image_tensor = image_tensor.unsqueeze(0)

    else:
        raise ValueError(f"Unknown format type: {format_type}")

    return image_tensor

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
