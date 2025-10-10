import viser
from nerfview import Viewer, RenderTabState
from typing import Literal
from typing import Callable, Tuple


class GaussianRenderTabState(RenderTabState):
    # non-controlable parameters
    total_count_number: int = 0
    rendered_count_number: int = 0
    fps: float = 0.0  # Frames per second
    _fps_smoothed: float = 0.0  # Internal smoothed FPS for display
    _fps_alpha: float = 0.1  # EMA smoothing factor (lower = smoother)

    # controlable parameters
    near_plane: float = 1e-3
    far_plane: float = 1e3
    radius_clip: float = 0.0  # 2D radius clip for rendering
    opacity_threshold: float = 0.005  # Minimum opacity for rendering (only used when percentile is off)
    opacity_percentile: float = 0.0  # Show top X% most opaque Gaussians (0 = all, 90 = top 10%)
    use_opacity_percentile: bool = True  # Use percentile instead of absolute threshold (default: True)
    scale_threshold: float = 100.0  # Maximum scale for rendering
    x_threshold: float = float('inf')  # X-axis threshold for cutting plane
    backgrounds: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    render_mode: Literal[
        "RGB", "Alpha", "Diffuse", "Specular", "Depth", "Normal"
    ] = "RGB"
    color_interpolation: float = 0.0  # 0.0 = SH_0 only, 1.0 = SH_1 only, 0.5 = 50/50 blend
    tight_snugbox: bool = True  # Use tight snugbox for TCGS rasterization (default: True)


class GaussianViewer(Viewer):
    def __init__(
        self,
        server: viser.ViserServer,
        render_fn: Callable,
        input_dim: int = 6,
        mode: Literal["rendering", "training"] = "rendering",
        share_url: bool = False,
        scene_bounds: Tuple[float, float] = None,  # (x_min, x_max) for x_threshold slider
    ):
        self.input_dim = input_dim
        self.scene_bounds = scene_bounds
        super().__init__(server, render_fn, mode=mode)
        # Configure the panel
        server.gui.set_panel_label("6D Gaussian Splatting Viewer")
        server.gui.configure_theme(control_width="large")
        if share_url:
            server.request_share_url()

    def _init_rendering_tab(self):
        self.render_tab_state = GaussianRenderTabState()
        self._rendering_tab_handles = {}
        self._rendering_folder = self.server.gui.add_folder("Rendering")

    def _populate_rendering_tab(self):
        with self._rendering_folder:
            # FPS display at the very top
            self.fps_number = self.server.gui.add_number(
                "FPS",
                initial_value=self.render_tab_state.fps,
                disabled=True,
                hint="Frames per second (rendering performance)",
            )

            with self.server.gui.add_folder("Gaussian Filtering"):
                # Toggle between absolute threshold and percentile
                self.use_opacity_percentile_checkbox = self.server.gui.add_checkbox(
                    "Use Opacity Percentile",
                    initial_value=self.render_tab_state.use_opacity_percentile,
                    hint="Use percentile filtering instead of absolute threshold",
                )

                @self.use_opacity_percentile_checkbox.on_update
                def _(_) -> None:
                    self.render_tab_state.use_opacity_percentile = self.use_opacity_percentile_checkbox.value
                    # Toggle slider visibility
                    self.opacity_threshold_slider.disabled = self.use_opacity_percentile_checkbox.value
                    self.opacity_percentile_slider.disabled = not self.use_opacity_percentile_checkbox.value
                    self.rerender(_)

                # Absolute threshold slider
                self.opacity_threshold_slider = self.server.gui.add_slider(
                    "Opacity Threshold",
                    min=0.0,
                    max=1.0,
                    step=0.001,
                    initial_value=self.render_tab_state.opacity_threshold,
                    hint="Minimum opacity for rendering Gaussians (0.0 = show all, 0.5 = half transparent+, 1.0 = fully opaque only)",
                    disabled=self.render_tab_state.use_opacity_percentile,
                )

                @self.opacity_threshold_slider.on_update
                def _(_) -> None:
                    if not self.render_tab_state.use_opacity_percentile:
                        self.render_tab_state.opacity_threshold = self.opacity_threshold_slider.value
                        self.rerender(_)

                # Percentile slider
                self.opacity_percentile_slider = self.server.gui.add_slider(
                    "Opacity Percentile",
                    min=0.0,
                    max=100.0,
                    step=0.1,
                    initial_value=self.render_tab_state.opacity_percentile,
                    hint="Show top X% most opaque Gaussians (0 = all, 50 = top 50%, 90 = top 10%, 99 = top 1%)",
                    disabled=not self.render_tab_state.use_opacity_percentile,
                )

                @self.opacity_percentile_slider.on_update
                def _(_) -> None:
                    if self.render_tab_state.use_opacity_percentile:
                        self.render_tab_state.opacity_percentile = self.opacity_percentile_slider.value
                        self.rerender(_)

                self.scale_threshold_slider = self.server.gui.add_slider(
                    "Scale Threshold",
                    min=0.1,
                    max=200.0,
                    step=0.1,
                    initial_value=self.render_tab_state.scale_threshold,
                    hint="Maximum scale for rendering Gaussians",
                )

                @self.scale_threshold_slider.on_update
                def _(_) -> None:
                    self.render_tab_state.scale_threshold = self.scale_threshold_slider.value
                    self.rerender(_)

            with self.server.gui.add_folder("Cutting Plane"):
                # Calculate x_threshold range from scene bounds
                if self.scene_bounds is not None:
                    x_min, x_max = self.scene_bounds
                    # Add 20% margin on each side
                    x_range = x_max - x_min
                    margin = x_range * 0.2
                    slider_min = round(x_min - margin, 1)  # Round to 1 decimal for cleaner UI
                    slider_max = round(x_max + margin, 1)  # Round to 1 decimal for cleaner UI
                    slider_initial = round((x_min + x_max) / 2.0, 1)  # Center of scene
                    slider_step = round(x_range / 1000.0, 3)  # 1000 steps across range
                else:
                    # Fallback to default values
                    slider_min = -100.0
                    slider_max = 100.0
                    slider_initial = 0.0
                    slider_step = 0.1

                self.x_threshold_checkbox = self.server.gui.add_checkbox(
                    "Enable X Threshold",
                    initial_value=False,
                    hint="Enable/disable cutting plane",
                )

                self.x_threshold_slider = self.server.gui.add_slider(
                    "X Threshold",
                    min=slider_min,
                    max=slider_max,
                    step=slider_step,
                    initial_value=slider_initial,
                    hint=f"X-axis threshold for cutting plane (range: {slider_min:.1f} to {slider_max:.1f})",
                    disabled=True,  # Disabled by default until checkbox is enabled
                )

                @self.x_threshold_checkbox.on_update
                def _(_) -> None:
                    # Enable/disable the slider based on checkbox
                    self.x_threshold_slider.disabled = not self.x_threshold_checkbox.value

                    if self.x_threshold_checkbox.value:
                        self.render_tab_state.x_threshold = self.x_threshold_slider.value
                    else:
                        self.render_tab_state.x_threshold = float('inf')
                    self.rerender(_)

                @self.x_threshold_slider.on_update
                def _(_) -> None:
                    # Only update if checkbox is enabled
                    if self.x_threshold_checkbox.value:
                        self.render_tab_state.x_threshold = self.x_threshold_slider.value
                        self.rerender(_)
            
            with self.server.gui.add_folder("Color Interpolation (Dual SH)"):
                self.color_interpolation_slider = self.server.gui.add_slider(
                    "Color Blend",
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    initial_value=self.render_tab_state.color_interpolation,
                    hint="Interpolate between two SH colors: 0.0 = Color 0, 0.5 = 50/50 blend, 1.0 = Color 1",
                )

                @self.color_interpolation_slider.on_update
                def _(_) -> None:
                    self.render_tab_state.color_interpolation = self.color_interpolation_slider.value
                    self.rerender(_)

            with self.server.gui.add_folder("Render Mode"):
                self.render_mode_dropdown = self.server.gui.add_dropdown(
                    "Mode",
                    ["RGB", "Alpha", "Diffuse", "Specular", "Depth", "Normal"],
                    initial_value=self.render_tab_state.render_mode,
                )

                @self.render_mode_dropdown.on_update
                def _(_) -> None:
                    self.render_tab_state.render_mode = self.render_mode_dropdown.value
                    self.rerender(_)

                self.tight_snugbox_checkbox = self.server.gui.add_checkbox(
                    "Tight Snugbox",
                    initial_value=self.render_tab_state.tight_snugbox,
                    hint="Use tight snugbox for TCGS rasterization (better culling, may improve performance)",
                )

                @self.tight_snugbox_checkbox.on_update
                def _(_) -> None:
                    self.render_tab_state.tight_snugbox = self.tight_snugbox_checkbox.value
                    self.rerender(_)

                self.total_count_number = self.server.gui.add_number(
                    "Total",
                    initial_value=self.render_tab_state.total_count_number,
                    disabled=True,
                    hint="Total number of splats in the scene.",
                )
                self.rendered_count_number = self.server.gui.add_number(
                    "Rendered",
                    initial_value=self.render_tab_state.rendered_count_number,
                    disabled=True,
                    hint="Number of splats rendered.",
                )
                self.radius_clip_slider = self.server.gui.add_number(
                    "Radius Clip",
                    initial_value=self.render_tab_state.radius_clip,
                    min=0.0,
                    max=100.0,
                    step=1.0,
                    hint="2D radius clip for rendering.",
                )

                @self.radius_clip_slider.on_update
                def _(_) -> None:
                    self.render_tab_state.radius_clip = self.radius_clip_slider.value
                    self.rerender(_)

                self.near_far_plane_vec2 = self.server.gui.add_vector2(
                    "Near/Far",
                    initial_value=(
                        self.render_tab_state.near_plane,
                        self.render_tab_state.far_plane,
                    ),
                    min=(1e-3, 1e1),
                    max=(1e1, 1e3),
                    step=1e-3,
                    hint="Near and far plane for rendering.",
                )

                @self.near_far_plane_vec2.on_update
                def _(_) -> None:
                    (
                        self.render_tab_state.near_plane,
                        self.render_tab_state.far_plane,
                    ) = self.near_far_plane_vec2.value
                    self.rerender(_)

                self.backgrounds_slider = self.server.gui.add_rgb(
                    "Background",
                    initial_value=self.render_tab_state.backgrounds,
                    hint="Background color for rendering.",
                )

                @self.backgrounds_slider.on_update
                def _(_) -> None:
                    self.render_tab_state.backgrounds = self.backgrounds_slider.value
                    self.rerender(_)

        self._rendering_tab_handles.update(
            {
                "use_opacity_percentile_checkbox": self.use_opacity_percentile_checkbox,
                "opacity_threshold_slider": self.opacity_threshold_slider,
                "opacity_percentile_slider": self.opacity_percentile_slider,
                "scale_threshold_slider": self.scale_threshold_slider,
                "x_threshold_slider": self.x_threshold_slider,
                "x_threshold_checkbox": self.x_threshold_checkbox,
                "color_interpolation_slider": self.color_interpolation_slider,
                "tight_snugbox_checkbox": self.tight_snugbox_checkbox,
                "fps_number": self.fps_number,
                "total_count_number": self.total_count_number,
                "rendered_count_number": self.rendered_count_number,
                "near_far_plane_vec2": self.near_far_plane_vec2,
                "radius_clip_slider": self.radius_clip_slider,
                "render_mode_dropdown": self.render_mode_dropdown,
                "backgrounds_slider": self.backgrounds_slider,
            }
        )
        super()._populate_rendering_tab()

    def _after_render(self):
        # Update the GUI elements with current values
        # Apply exponential moving average to smooth FPS display
        if self.render_tab_state._fps_smoothed == 0.0:
            # Initialize on first frame
            self.render_tab_state._fps_smoothed = self.render_tab_state.fps
        else:
            # EMA: smoothed = alpha * new + (1 - alpha) * old
            self.render_tab_state._fps_smoothed = (
                self.render_tab_state._fps_alpha * self.render_tab_state.fps +
                (1.0 - self.render_tab_state._fps_alpha) * self.render_tab_state._fps_smoothed
            )

        self._rendering_tab_handles[
            "fps_number"
        ].value = round(self.render_tab_state._fps_smoothed, 1)
        self._rendering_tab_handles[
            "total_count_number"
        ].value = self.render_tab_state.total_count_number
        self._rendering_tab_handles[
            "rendered_count_number"
        ].value = self.render_tab_state.rendered_count_number
