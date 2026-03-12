import time
import torch
import viser
import numpy as np
from argparse import ArgumentParser
from arguments import ModelParams, ViewerParams
from scene import get_gaussian_model
from scene.gaussian_viewer import GaussianViewer
from scene.beta_viewer import BetaViewer


@torch.no_grad()
def viewing(model_params, viewer_params, ply_path, input_dim=6, auto_camera=True, share_url=False):
    """Launch interactive viewer for trained NDGS models.

    Args:
        model_params: ModelParams object containing model settings
        viewer_params: ViewerParams object containing viewer settings
        ply_path: Path to the .ply model file
        input_dim: Input dimension for UBS model
        auto_camera: Whether to automatically position camera to look at object center (default: True)
        share_url: Whether to share viewer URL
    """
    # Get the appropriate GaussianModel class based on mode
    GaussianModel = get_gaussian_model(model_params.mode)

    # Initialize model
    if "ubs" in model_params.mode:
        gaussian_model = GaussianModel(input_dim=input_dim)
    elif "ndgs" in model_params.mode:
        gaussian_model = GaussianModel(model_params.sh_degree, input_dim=input_dim,
                                       use_rot_scale_l_triangle=model_params.use_rot_scale_l_triangle)
    elif "dgs" in model_params.mode:
        gaussian_model = GaussianModel(
            model_params.sh_degree,
            input_dim=input_dim,
            use_beta=model_params.use_beta,
            use_view_dependent_pos=model_params.use_view_dependent_pos,
            use_opacity_pos_decouple=model_params.use_opacity_pos_decouple,
            l_22_inv_init_scale=model_params.l_22_inv_init_scale,
            lambda_init=model_params.lambda_init
        )
    else:
        gaussian_model = GaussianModel(model_params.sh_degree)

    # Load model from file
    print(f"Loading model from {ply_path}")
    gaussian_model.load_ply(ply_path)

    # Set background color
    bg_color = [1, 1, 1] if model_params.white_background else [0, 0, 0]
    gaussian_model.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Compute scene center and bounds for camera setup
    xyz = gaussian_model.get_xyz.detach().cpu().numpy()
    scene_center = xyz.mean(axis=0)
    scene_min = xyz.min(axis=0)
    scene_max = xyz.max(axis=0)
    scene_radius = np.linalg.norm(scene_max - scene_min) / 2.0

    # Scene bounds for cutting plane slider
    x_min, x_max = xyz[:, 0].min(), xyz[:, 0].max()
    scene_bounds = (float(x_min), float(x_max))

    # Start viser server
    server = viser.ViserServer(port=viewer_params.port, verbose=False)

    # Set up camera callback to position camera nicely
    if auto_camera:
        @server.on_client_connect
        def on_connect(client):
            camera_distance = scene_radius * 1.2  # Add 20% margin for better view

            # Position camera slightly above and in front of the scene
            # This gives a nice 3/4 view of the object
            initial_camera_position = scene_center + np.array([
                camera_distance * 0.5,   # Slightly to the right
                camera_distance * 0.3,   # Slightly above
                camera_distance          # In front
            ])

            # Set camera position and look_at target
            client.camera.position = tuple(initial_camera_position)
            client.camera.look_at = tuple(scene_center)
            client.camera.up_direction = (0.0, 1.0, 0.0)

    # Create viewer - Use BetaViewer for UBS mode, GaussianViewer for others
    ViewerClass = BetaViewer if "ubs" in model_params.mode else GaussianViewer
    viewer = ViewerClass(
        server=server,
        render_fn=lambda camera_state, render_tab_state: gaussian_model.view_tcgs(
            camera_state, render_tab_state
        ),
        input_dim=input_dim,
        mode="rendering",
        share_url=share_url,
        scene_bounds=scene_bounds,
    )

    print(f"Viewer running on http://localhost:{viewer_params.port}")
    print("Ctrl+C to exit.")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nShutting down viewer...")


if __name__ == "__main__":
    parser = ArgumentParser(description="NDGS Viewing script")
    model_params = ModelParams(parser)
    viewer_params = ViewerParams(parser)

    # Add view-specific arguments (avoid duplicates with ModelParams)
    parser.add_argument("--ply", type=str, required=True, help="Path to the .ply file")
    parser.add_argument(
        "--share_url",
        action="store_true",
        help="Share URL for the viewer (requires viser)",
    )
    parser.add_argument(
        "--auto_camera",
        action="store_true",
        default=True,
        help="Automatically position camera to view object (default: True, use --no-auto-camera to disable)",
    )
    parser.add_argument(
        "--no-auto-camera",
        dest="auto_camera",
        action="store_false",
        help="Do not automatically position camera",
    )

    args = parser.parse_args()

    # Extract parameters
    model = model_params.extract(args)
    viewer = viewer_params.extract(args)

    viewing(model, viewer, args.ply, args.input_dim, args.auto_camera, args.share_url)
