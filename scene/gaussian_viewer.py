import viser
from nerfview import Viewer, RenderTabState
from typing import Literal
from typing import Callable, Tuple


class GaussianRenderTabState(RenderTabState):
    # non-controlable parameters
    total_count_number: int = 0
    rendered_count_number: int = 0

    # controlable parameters
    near_plane: float = 1e-3
    far_plane: float = 1e3
    radius_clip: float = 0.0  # 2D radius clip for rendering
    opacity_threshold: float = 0.005  # Minimum opacity for rendering (default very low to show most Gaussians)
    scale_threshold: float = 100.0  # Maximum scale for rendering
    x_threshold: float = float('inf')  # X-axis threshold for cutting plane
    backgrounds: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    render_mode: Literal[
        "RGB", "Alpha", "Diffuse", "Specular", "Depth", "Normal"
    ] = "RGB"


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
        server.gui.set_panel_label("6D Gaussian Splatting Viewer")
        if share_url:
            server.request_share_url()

    def _init_rendering_tab(self):
        self.render_tab_state = GaussianRenderTabState()
        self._rendering_tab_handles = {}
        self._rendering_folder = self.server.gui.add_folder("Rendering")

    def _populate_rendering_tab(self):
        with self._rendering_folder:

            with self.server.gui.add_folder("Gaussian Filtering"):
                self.opacity_threshold_slider = self.server.gui.add_slider(
                    "Opacity Threshold",
                    min=0.0,
                    max=0.5,
                    step=0.001,
                    initial_value=self.render_tab_state.opacity_threshold,
                    hint="Minimum opacity for rendering Gaussians (0.005 = show most, 0.1 = only opaque)",
                )

                @self.opacity_threshold_slider.on_update
                def _(_) -> None:
                    self.render_tab_state.opacity_threshold = self.opacity_threshold_slider.value
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
                    slider_min = x_min - margin
                    slider_max = x_max + margin
                    slider_initial = (x_min + x_max) / 2.0  # Center of scene
                    slider_step = x_range / 1000.0  # 1000 steps across range
                else:
                    # Fallback to default values
                    slider_min = -100.0
                    slider_max = 100.0
                    slider_initial = 0.0
                    slider_step = 0.1

                self.x_threshold_slider = self.server.gui.add_slider(
                    "X Threshold",
                    min=slider_min,
                    max=slider_max,
                    step=slider_step,
                    initial_value=slider_initial,
                    hint=f"X-axis threshold for cutting plane (range: {slider_min:.2f} to {slider_max:.2f})",
                )

                @self.x_threshold_slider.on_update
                def _(_) -> None:
                    self.render_tab_state.x_threshold = self.x_threshold_slider.value
                    self.rerender(_)

                self.x_threshold_checkbox = self.server.gui.add_checkbox(
                    "Enable X Threshold",
                    initial_value=False,
                    hint="Enable/disable cutting plane",
                )

                @self.x_threshold_checkbox.on_update
                def _(_) -> None:
                    if self.x_threshold_checkbox.value:
                        self.render_tab_state.x_threshold = self.x_threshold_slider.value
                    else:
                        self.render_tab_state.x_threshold = float('inf')
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
                "opacity_threshold_slider": self.opacity_threshold_slider,
                "scale_threshold_slider": self.scale_threshold_slider,
                "x_threshold_slider": self.x_threshold_slider,
                "x_threshold_checkbox": self.x_threshold_checkbox,
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
        self._rendering_tab_handles[
            "total_count_number"
        ].value = self.render_tab_state.total_count_number
        self._rendering_tab_handles[
            "rendered_count_number"
        ].value = self.render_tab_state.rendered_count_number
