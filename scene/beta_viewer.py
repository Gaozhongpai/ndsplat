import viser
from nerfview import Viewer, RenderTabState
from typing import Literal
from typing import Callable, Tuple


class BetaRenderTabState(RenderTabState):
    # non-controlable parameters
    total_count_number: int = 0
    rendered_count_number: int = 0
    fps: float = 0.0  # Frames per second
    _fps_smoothed: float = 0.0  # Internal smoothed FPS for display
    _fps_alpha: float = 0.1  # EMA smoothing factor (lower = smoother)

    # controlable parameters
    timestamp: float = 0.0
    near_plane: float = 1e-3
    far_plane: float = 1e3
    radius_clip: float = 0.0
    x_threshold: float = float('inf')  # X-axis threshold for cutting plane
    b_xyz: Tuple[int, int] = (0, 100)
    b_view: Tuple[int, int] = (0, 100)
    b_time: Tuple[int, int] = (0, 100)
    backgrounds: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    render_mode: Literal[
        "RGB", "Alpha", "Diffuse", "Specular", "Depth", "Normal"
    ] = "RGB"


class BetaViewer(Viewer):
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
        panel_label = f"{input_dim}D Beta Splatting Viewer"
        if input_dim == 7:
            panel_label += " (with Time)"
        server.gui.set_panel_label(panel_label)
        server.gui.configure_theme(
            control_width="large",
            dark_mode=True,
            brand_color=(255, 211, 105),
        )
        if share_url:
            server.request_share_url()

    def _init_rendering_tab(self):
        self.render_tab_state = BetaRenderTabState()
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

            if self.input_dim == 7:
                self.gui_slider_time = self.server.gui.add_slider(
                    "Time",
                    min=0.0,
                    max=1.0,
                    step=0.0001,
                    initial_value=self.render_tab_state.timestamp,
                )

                @self.gui_slider_time.on_update
                def _(_) -> None:
                    self.render_tab_state.timestamp = self.gui_slider_time.value
                    self.rerender(_)

            with self.server.gui.add_folder("Geometry Dependency Control"):
                self.gui_multi_slider_xyz = self.server.gui.add_multi_slider(
                    "Geo quantile",
                    min=0,
                    max=100,
                    step=1,
                    initial_value=self.render_tab_state.b_xyz,
                )

                @self.gui_multi_slider_xyz.on_update
                def _(_) -> None:
                    self.render_tab_state.b_xyz = self.gui_multi_slider_xyz.value
                    self.rerender(_)

            with self.server.gui.add_folder("View Dependency Control"):
                self.gui_multi_slider_view = self.server.gui.add_multi_slider(
                    "View quantile",
                    min=0,
                    max=100,
                    step=1,
                    initial_value=self.render_tab_state.b_view,
                )

                @self.gui_multi_slider_view.on_update
                def _(_) -> None:
                    self.render_tab_state.b_view = self.gui_multi_slider_view.value
                    self.rerender(_)

            if self.input_dim == 7:
                with self.server.gui.add_folder("Time Dependency Control"):
                    self.gui_multi_slider_time = self.server.gui.add_multi_slider(
                        "Time quantile",
                        min=0,
                        max=100,
                        step=1,
                        initial_value=self.render_tab_state.b_time,
                    )

                    @self.gui_multi_slider_time.on_update
                    def _(_) -> None:
                        self.render_tab_state.b_time = self.gui_multi_slider_time.value
                        self.rerender(_)

            with self.server.gui.add_folder("Cutting Plane"):
                # Calculate x_threshold range from scene bounds
                if self.scene_bounds is not None:
                    x_min, x_max = self.scene_bounds
                    # Add 20% margin on each side
                    x_range = x_max - x_min
                    margin = x_range * 0.2
                    slider_min = round(x_min - margin, 1)
                    slider_max = round(x_max + margin, 1)
                    slider_initial = round((x_min + x_max) / 2.0, 1)
                    slider_step = round(x_range / 1000.0, 3)
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
                "timestamp": self.gui_slider_time if self.input_dim == 7 else 0.0,
                "b_xyz": self.gui_multi_slider_xyz,
                "b_view": self.gui_multi_slider_view,
                "b_time": self.gui_multi_slider_time
                if self.input_dim == 7
                else (0, 100),
                "x_threshold_checkbox": self.x_threshold_checkbox,
                "x_threshold_slider": self.x_threshold_slider,
                "fps_number": self.fps_number,
                "total_count_number": self.total_count_number,
                "rendered_count_number": self.rendered_count_number,
                "near_far_plane_vec2": self.near_far_plane_vec2,
                "radius_clip_slider": self.radius_clip_slider,
                "rener_mode_dropdown": self.render_mode_dropdown,
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
