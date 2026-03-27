"""
Panda3D scene manager with real-world CC0 textures from Poly Haven.
Textures are downloaded once and cached in assets/textures/.
Road tiles are recycled as the vehicle advances for infinite road generation.
"""
import os
import math
import random
import logging
import urllib.request
import json
from typing import List, Optional

import numpy as np
from panda3d.core import GraphicsOutput

log = logging.getLogger(__name__)

_ASSET_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "textures")
)
_UA = {"User-Agent": "SelfDrivingSimulator/1.0 (github research project)"}

# Verified working Poly Haven direct download URLs (CC0)
TEXTURE_URLS = {
    "asphalt":  "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/asphalt_02/asphalt_02_diff_1k.jpg",
    "road2":    "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/aerial_asphalt_01/aerial_asphalt_01_diff_1k.jpg",
    "ground":   "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/brown_mud_leaves_01/brown_mud_leaves_01_diff_1k.jpg",
    "grass":    "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/grass_path_2/grass_path_2_diff_1k.jpg",
    "concrete": "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/concrete_wall_001/concrete_wall_001_diff_1k.jpg",
    "sky":      "https://dl.polyhaven.org/file/ph-assets/HDRIs/extra/Tonemapped%20JPG/kloofendal_48d_partly_cloudy_puresky.jpg",
}


def _download_textures() -> dict:
    """
    Download real-world CC0 textures from Poly Haven.
    Cached in assets/textures/ — only downloads once.
    Returns dict of name -> local file path.
    """
    os.makedirs(_ASSET_DIR, exist_ok=True)
    paths = {}
    for name, url in TEXTURE_URLS.items():
        dest = os.path.join(_ASSET_DIR, f"{name}.jpg")
        paths[name] = dest
        if os.path.exists(dest) and os.path.getsize(dest) > 10_000:
            log.info("Texture cached: %s", name)
            continue
        log.info("Downloading real-world texture: %s ...", name)
        try:
            req = urllib.request.Request(url, headers=_UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read()
            with open(dest, "wb") as f:
                f.write(data)
            log.info("  -> %s (%.1f KB)", name, len(data) / 1024)
        except Exception as e:
            log.warning("  -> FAILED: %s — will use fallback", e)
            paths[name] = None
    return paths


def _load_panda_texture(loader, path: Optional[str], fallback_rgb=(128, 128, 128)):
    """Load a texture from disk, or create a solid-colour fallback."""
    from panda3d.core import Texture, Filename
    if path and os.path.exists(path) and os.path.getsize(path) > 10_000:
        # Use Panda3D's Filename to handle Windows paths correctly
        panda_path = Filename.from_os_specific(path)
        tex = loader.load_texture(panda_path)
        if tex:
            tex.set_wrap_u(Texture.WMRepeat)
            tex.set_wrap_v(Texture.WMRepeat)
            tex.set_minfilter(Texture.FTLinearMipmapLinear)
            tex.set_magfilter(Texture.FTLinear)
            return tex
    # Solid colour fallback (4x4 pixels)
    r, g, b = fallback_rgb
    arr = np.full((4, 4, 3), [r, g, b], dtype=np.uint8)
    tex = Texture("fallback")
    tex.setup_2d_texture(4, 4, Texture.TUnsignedByte, Texture.FRgb)
    tex.set_ram_image(bytes(arr.tobytes()))
    return tex


class SceneManager:
    def __init__(self, base, config):
        self.base = base
        self.config = config
        self.tile_length = config.road_tile_length
        self.road_width = config.road_width
        self.tiles_ahead = config.road_tiles_ahead

        # Download real textures (cached after first run)
        tex_paths = _download_textures()

        # Load Panda3D textures
        self._tex = {
            "asphalt":  _load_panda_texture(base.loader, tex_paths.get("asphalt"),  (60, 60, 60)),
            "road2":    _load_panda_texture(base.loader, tex_paths.get("road2"),    (55, 55, 55)),
            "ground":   _load_panda_texture(base.loader, tex_paths.get("ground"),   (90, 70, 50)),
            "grass":    _load_panda_texture(base.loader, tex_paths.get("grass"),    (50, 110, 40)),
            "concrete": _load_panda_texture(base.loader, tex_paths.get("concrete"), (180, 175, 165)),
            "sky":      _load_panda_texture(base.loader, tex_paths.get("sky"),      (135, 200, 235)),
        }

        self._road_tiles: List = []
        self._next_tile_y: float = 0.0
        self._haze_effect = None

        self._setup_lighting()
        self._setup_sky()
        self._setup_ground_plane()
        self._init_road_pool()
        self._setup_camera()

    # ------------------------------------------------------------------
    # Lighting
    # ------------------------------------------------------------------

    def _setup_lighting(self) -> None:
        from panda3d.core import DirectionalLight, AmbientLight, Vec4
        dlight = DirectionalLight("sun")
        dlight.set_color(Vec4(1.0, 0.95, 0.85, 1))
        dlnp = self.base.render.attach_new_node(dlight)
        dlnp.set_hpr(45, -60, 0)
        self.base.render.set_light(dlnp)
        alight = AmbientLight("ambient")
        alight.set_color(Vec4(0.4, 0.4, 0.45, 1))
        alnp = self.base.render.attach_new_node(alight)
        self.base.render.set_light(alnp)

    # ------------------------------------------------------------------
    # Sky — large textured backdrop that follows the camera
    # ------------------------------------------------------------------

    def _setup_sky(self) -> None:
        from panda3d.core import CardMaker, TextureStage
        cm = CardMaker("sky_bg")
        cm.set_frame(-600, 600, -50, 300)
        sky_np = self.base.render.attach_new_node(cm.generate())
        sky_np.set_pos(0, 800, 0)
        sky_np.set_bin("background", 0)
        sky_np.set_depth_write(False)
        sky_np.set_compass()
        sky_np.set_texture(self._tex["sky"])
        sky_np.set_light_off()   # sky should not be affected by scene lights

    # ------------------------------------------------------------------
    # Wide ground plane (dirt/mud outside road)
    # ------------------------------------------------------------------

    def _setup_ground_plane(self) -> None:
        from panda3d.core import CardMaker, TextureStage
        cm = CardMaker("ground_plane")
        cm.set_frame(-200, 200, -500, 5000)
        gnd = self.base.render.attach_new_node(cm.generate())
        gnd.set_p(-90)
        gnd.set_pos(0, 0, -0.02)   # just below road tiles
        gnd.set_texture(self._tex["ground"])
        gnd.set_tex_scale(TextureStage.get_default(), 20, 200)

    # ------------------------------------------------------------------
    # Road tile pool
    # ------------------------------------------------------------------

    def _init_road_pool(self) -> None:
        for _ in range(self.tiles_ahead + 2):
            self._spawn_tile(self._next_tile_y)
            self._next_tile_y += self.tile_length

    def _spawn_tile(self, y_pos: float):
        from panda3d.core import CardMaker, TextureStage
        w = self.road_width
        l = self.tile_length

        # Alternate between two asphalt textures for variety
        tex_key = "asphalt" if int(y_pos / l) % 2 == 0 else "road2"

        cm = CardMaker(f"road_{y_pos:.0f}")
        cm.set_frame(-w / 2, w / 2, 0, l)
        tile_np = self.base.render.attach_new_node(cm.generate())
        tile_np.set_p(-90)
        tile_np.set_pos(0, y_pos, 0)
        tile_np.set_texture(self._tex[tex_key])
        tile_np.set_tex_scale(TextureStage.get_default(), 1.5, l / 4)

        # Concrete kerb strips on each side
        self._add_kerbs(tile_np, y_pos, w, l)
        # Dashed white lane markings
        self._add_lane_markings(tile_np, y_pos, w, l)
        # Roadside vegetation / trees
        self._add_vegetation(y_pos, l)

        self._road_tiles.append(tile_np)
        return tile_np

    def _add_kerbs(self, parent, y_pos: float, w: float, l: float) -> None:
        """Concrete kerb strips along road edges."""
        from panda3d.core import CardMaker, TextureStage
        for side in [-1, 1]:
            x = side * (w / 2 + 0.3)
            cm = CardMaker(f"kerb_{y_pos:.0f}_{side}")
            cm.set_frame(-0.3, 0.3, 0, l)
            k = parent.attach_new_node(cm.generate())
            k.set_pos(x, 0, 0.01)
            k.set_texture(self._tex["concrete"])
            k.set_tex_scale(TextureStage.get_default(), 1, l / 2)

    def _add_lane_markings(self, parent, y_pos: float, w: float, l: float) -> None:
        """Dashed centre and lane lines."""
        from panda3d.core import CardMaker
        # Centre dashed line
        dash_len = 3.0
        gap_len  = 3.0
        y = 0.0
        while y < l:
            cm = CardMaker(f"dash_{y_pos:.0f}_{y:.0f}")
            cm.set_frame(-0.1, 0.1, y, min(y + dash_len, l))
            d = parent.attach_new_node(cm.generate())
            d.set_pos(0, 0, 0.015)
            d.set_color(1, 1, 0.8, 1)   # warm white/yellow
            y += dash_len + gap_len

        # Solid edge lines
        for x_off in [-w * 0.45, w * 0.45]:
            cm = CardMaker(f"edge_{y_pos:.0f}_{x_off:.1f}")
            cm.set_frame(-0.08, 0.08, 0, l)
            e = parent.attach_new_node(cm.generate())
            e.set_pos(x_off, 0, 0.015)
            e.set_color(1, 1, 1, 1)

    def _add_vegetation(self, y_pos: float, l: float) -> None:
        """Billboard grass patches and tree trunks along road edges."""
        from panda3d.core import CardMaker, TextureStage
        rng = random.Random(int(y_pos * 137))

        for side in [-1, 1]:
            x_base = side * (self.road_width / 2 + 1.5)

            # Grass ground patches (flat quads)
            for i in range(4):
                cm = CardMaker(f"gp_{y_pos:.0f}_{side}_{i}")
                w_g = rng.uniform(3, 7)
                cm.set_frame(-w_g, w_g, 0, rng.uniform(4, 10))
                gp = self.base.render.attach_new_node(cm.generate())
                gp.set_p(-90)
                gp.set_pos(x_base + side * rng.uniform(0, 8),
                           y_pos + rng.uniform(0, l), -0.01)
                gp.set_texture(self._tex["grass"])
                gp.set_tex_scale(TextureStage.get_default(), 2, 3)

            # Vertical billboard trees
            for i in range(3):
                cm = CardMaker(f"tree_{y_pos:.0f}_{side}_{i}")
                h_t = rng.uniform(4, 9)
                cm.set_frame(-h_t * 0.4, h_t * 0.4, 0, h_t)
                tree = self.base.render.attach_new_node(cm.generate())
                tree.set_pos(x_base + side * rng.uniform(2, 12),
                             y_pos + rng.uniform(0, l), 0)
                tree.set_billboard_point_eye()
                tree.set_texture(self._tex["grass"])   # green billboard
                tree.set_color(0.4, 0.7, 0.3, 1)      # tint green
                tree.set_transparency(1)

    # ------------------------------------------------------------------
    # Tile recycling
    # ------------------------------------------------------------------

    def update(self, vehicle_y: float) -> None:
        to_remove = [t for t in self._road_tiles
                     if t.get_y() + self.tile_length < vehicle_y - self.tile_length]
        for tile in to_remove:
            tile.remove_node()
            self._road_tiles.remove(tile)

        furthest_y = max((t.get_y() for t in self._road_tiles), default=0.0)
        while furthest_y < vehicle_y + self.tiles_ahead * self.tile_length:
            furthest_y += self.tile_length
            self._spawn_tile(furthest_y)
            self._next_tile_y = furthest_y + self.tile_length

    # ------------------------------------------------------------------
    # Camera capture for inference
    # ------------------------------------------------------------------

    def _setup_camera(self) -> None:
        from panda3d.core import FrameBufferProperties, WindowProperties, GraphicsPipe
        try:
            fb_props = FrameBufferProperties()
            fb_props.set_rgb_color(True)
            fb_props.set_depth_bits(16)
            win_props = WindowProperties.size(224, 224)
            self._offscreen = self.base.graphics_engine.make_output(
                self.base.pipe, "inference_cam", -100,
                fb_props, win_props,
                GraphicsPipe.BFRefuseWindow | GraphicsPipe.BFSizeTrackHost,
                self.base.win.get_gsg(), self.base.win,
            )
            if self._offscreen:
                from panda3d.core import Camera
                self._inf_cam = Camera("inf_cam")
                self._inf_cam_np = self.base.render.attach_new_node(self._inf_cam)
                self._inf_cam_np.set_pos(0, -2, 1.2)
                self._inf_cam_np.set_hpr(0, -5, 0)
                dr = self._offscreen.make_display_region()
                dr.set_camera(self._inf_cam_np)
                log.info("Offscreen inference camera created (224x224).")
            else:
                self._offscreen = None
        except Exception as e:
            log.warning("Offscreen camera setup failed: %s", e)
            self._offscreen = None

    def capture_frame(self) -> np.ndarray:
        """Capture current view as (224, 224, 3) uint8 BGR numpy array."""
        try:
            import cv2
            from panda3d.core import Texture
            if not hasattr(self, '_capture_tex'):
                self._capture_tex = Texture("capture")
                self.base.win.add_render_texture(
                    self._capture_tex,
                    GraphicsOutput.RTMCopyRam,
                    GraphicsOutput.RTPColor,
                )
            self.base.graphicsEngine.extract_texture_data(
                self._capture_tex, self.base.win.get_gsg()
            )
            data = self._capture_tex.get_ram_image_as("BGR")
            w = self._capture_tex.get_x_size()
            h = self._capture_tex.get_y_size()
            if w == 0 or h == 0 or len(data) == 0:
                return np.zeros((224, 224, 3), dtype=np.uint8)
            arr = np.frombuffer(bytes(data), dtype=np.uint8).reshape((h, w, 3))
            arr = np.flipud(arr)
            return cv2.resize(arr, (224, 224))
        except Exception as e:
            log.warning("Frame capture failed: %s — returning blank frame.", e)
            return np.zeros((224, 224, 3), dtype=np.uint8)

    def attach_vehicle_camera(self, vehicle_node) -> None:
        if hasattr(self, "_inf_cam_np"):
            self._inf_cam_np.reparent_to(vehicle_node)
            self._inf_cam_np.set_pos(0, -2, 1.2)
            self._inf_cam_np.set_hpr(0, -5, 0)

    # ------------------------------------------------------------------
    # Haze
    # ------------------------------------------------------------------

    def set_haze(self, active: bool, haze_effect=None) -> None:
        if haze_effect is None:
            return
        self._haze_effect = haze_effect
        if active:
            haze_effect.apply_to_scene(self.base.render)
        else:
            haze_effect.remove_from_scene(self.base.render)
