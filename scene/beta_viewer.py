import viser
from nerfview import Viewer, RenderTabState
from typing import Literal
from typing import Callable, Tuple


class BetaRenderTabState(RenderTabState):
    # non-controlable parameters
    total_count_number: int = 0
    rendered_count_number: int = 0

    # controlable parameters
    timestamp: float = 0.0
    near_plane: float = 1e-3
    far_plane: float = 1e3
    radius_clip: float = 0.0
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
    ):
        self.input_dim = input_dim
        super().__init__(server, render_fn, mode=mode)
        server.gui.set_panel_label("Beta Splatting Viewer")
        if share_url:
            server.request_share_url()

    def _init_rendering_tab(self):
        self.render_tab_state = BetaRenderTabState()
        self._rendering_tab_handles = {}
        self._rendering_folder = self.server.gui.add_folder("Rendering")

    def _populate_rendering_tab(self):
        with self._rendering_folder:

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
        self._rendering_tab_handles[
            "total_count_number"
        ].value = self.render_tab_state.total_count_number
        self._rendering_tab_handles[
            "rendered_count_number"
        ].value = self.render_tab_state.rendered_count_number
