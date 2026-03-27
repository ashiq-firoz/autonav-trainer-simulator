"""
Panda3D scene manager: road tile pool, sky, vegetation, lighting, camera capture.
Road tiles are recycled as the vehicle advances for infinite road generation.
"""
import os
import math
import random
import logging
from typing import List, Optional

import numpy as np

log = logging.getLogger(__name__)

# Texture download URLs (CC0 from ambientCG)
TEXTURE_URLS = {
    "asphalt_color":  "https://ambientcg.com/get?file=Asphalt012_1K-JPG_Color.jpg",
    "asphalt_normal": "https://ambientcg.com/get?file=Asphalt012_1K-JPG_NormalGL.jpg",
    "lane_marking":   "https://ambientcg.com/get?file=PaintedWood001_1K-JPG_Color.jpg",
    "sky":            "https://dl.polyhaven.org/file/ph-assets/HDRIs/jpg/1k/kloofendal_48d_partly_cloudy_1k.jpg",
    "grass":          "https://ambientcg.com/get?file=Ground054_1K-JPG_Color.jpg",
    "tree_billboard": "https://ambientcg.com/get?file=Bark007_1K-JPG_Color.jpg",
}


def _download_textures(asset_dir: str) -> None:
    """Download CC0 textures if not already present."""
    try:
        import urllib.request
        tex_dir = os.path.join(asset_dir, "textures")
        os.makedirs(tex_dir, exist_ok=True)
        for name, url in TEXTURE_URLS.items():
            dest = os.path.join(tex_dir, f"{name}.jpg")
            if not os.path.exists(dest):
                log.info("Downloading texture: %s", name)
                try:
                    urllib.request.urlretrieve(url, dest)
                except Exception as e:
                    log.warning("Could not download %s: %s — using fallback color.", name, e)
    except Exception as e:
        log.warning("Texture download failed: %s", e)


class SceneManager:
    def __init__(self, base, config):
        self.base = base
        self.config = config
        self.tile_length = config.road_tile_length
        self.road_width = config.road_width
        self.tiles_ahead = config.road_tiles_ahead

        self._asset_dir = os.path.join(os.path.dirname(__file__), "..", "assets")
        _download_textures(self._asset_dir)

        self._road_tiles: List = []
        self._next_tile_y: float = 0.0   # Y position of next tile to spawn
        self._haze_effect = None

        self._setup_lighting()
        self._setup_sky()
        self._init_road_pool()
        self._setup_camera()

    # ------------------------------------------------------------------
    # Lighting
    # ------------------------------------------------------------------

    def _setup_lighting(self) -> None:
        from panda3d.core import DirectionalLight, AmbientLight, Vec4, Vec3
        # Directional sun
        dlight = DirectionalLight("sun")
        dlight.set_color(Vec4(1.0, 0.95, 0.85, 1))
        dlnp = self.base.render.attach_new_node(dlight)
        dlnp.set_hpr(45, -60, 0)
        self.base.render.set_light(dlnp)
        # Ambient fill
        alight = AmbientLight("ambient")
        alight.set_color(Vec4(0.35, 0.35, 0.4, 1))
        alnp = self.base.render.attach_new_node(alight)
        self.base.render.set_light(alnp)

    # ------------------------------------------------------------------
    # Sky dome
    # ------------------------------------------------------------------

    def _setup_sky(self) -> None:
        from panda3d.core import CardMaker, TextureStage, Texture
        sky_path = os.path.join(self._asset_dir, "textures", "sky.jpg")
        cm = CardMaker("sky_bg")
        cm.set_frame(-200, 200, -10, 120)
        sky_np = self.base.render.attach_new_node(cm.generate())
        sky_np.set_pos(0, 500, 0)
        sky_np.set_bin("background", 0)
        sky_np.set_depth_write(False)
        sky_np.set_compass()
        if os.path.exists(sky_path):
            tex = self.base.loader.load_texture(sky_path)
            sky_np.set_texture(tex)
        else:
            sky_np.set_color(0.53, 0.81, 0.98, 1)  # fallback sky blue

    # ------------------------------------------------------------------
    # Road tile pool
    # ------------------------------------------------------------------

    def _init_road_pool(self) -> None:
        for _ in range(self.tiles_ahead + 2):
            self._spawn_tile(self._next_tile_y)
            self._next_tile_y += self.tile_length

    def _spawn_tile(self, y_pos: float):
        from panda3d.core import CardMaker, TextureStage, Texture, Vec3
        w = self.road_width
        l = self.tile_length

        # Road surface
        cm = CardMaker(f"road_{y_pos:.0f}")
        cm.set_frame(-w / 2, w / 2, 0, l)
        tile_np = self.base.render.attach_new_node(cm.generate())
        tile_np.set_p(-90)   # lay flat
        tile_np.set_pos(0, y_pos, 0)

        asphalt_path = os.path.join(self._asset_dir, "textures", "asphalt_color.jpg")
        if os.path.exists(asphalt_path):
            tex = self.base.loader.load_texture(asphalt_path)
            tex.set_wrap_u(Texture.WMRepeat)
            tex.set_wrap_v(Texture.WMRepeat)
            tile_np.set_texture(tex)
            tile_np.set_tex_scale(TextureStage.get_default(), 2, 4)
        else:
            tile_np.set_color(0.25, 0.25, 0.25, 1)  # dark grey fallback

        # Lane markings (dashed white lines)
        self._add_lane_markings(tile_np, y_pos, w, l)

        # Roadside vegetation
        self._add_vegetation(y_pos, l)

        self._road_tiles.append(tile_np)
        return tile_np

    def _add_lane_markings(self, parent, y_pos: float, w: float, l: float) -> None:
        from panda3d.core import CardMaker, TextureStage, Texture
        for x_offset in [-w * 0.1, w * 0.1]:
            cm = CardMaker(f"lane_{y_pos:.0f}_{x_offset:.1f}")
            cm.set_frame(-0.15, 0.15, 0, l)
            mark_np = parent.attach_new_node(cm.generate())
            mark_np.set_pos(x_offset, 0, 0.01)  # slightly above road
            mark_np.set_color(1, 1, 1, 1)

    def _add_vegetation(self, y_pos: float, l: float) -> None:
        from panda3d.core import CardMaker, Texture
        grass_path = os.path.join(self._asset_dir, "textures", "grass.jpg")
        for side in [-1, 1]:
            x_base = side * (self.road_width / 2 + 2)
            for i in range(3):
                cm = CardMaker(f"veg_{y_pos:.0f}_{side}_{i}")
                cm.set_frame(-1.5, 1.5, 0, 4)
                veg_np = self.base.render.attach_new_node(cm.generate())
                veg_np.set_pos(x_base + random.uniform(-1, 1),
                               y_pos + random.uniform(0, l),
                               0)
                veg_np.set_billboard_point_eye()
                if os.path.exists(grass_path):
                    tex = self.base.loader.load_texture(grass_path)
                    veg_np.set_texture(tex)
                    veg_np.set_transparency(1)
                else:
                    veg_np.set_color(0.2, 0.6, 0.2, 1)

    # ------------------------------------------------------------------
    # Tile recycling
    # ------------------------------------------------------------------

    def update(self, vehicle_y: float) -> None:
        """Recycle tiles behind vehicle and spawn new ones ahead."""
        # Remove tiles more than 1 tile behind vehicle
        to_remove = []
        for tile in self._road_tiles:
            tile_y = tile.get_y()
            if tile_y + self.tile_length < vehicle_y - self.tile_length:
                to_remove.append(tile)

        for tile in to_remove:
            tile.remove_node()
            self._road_tiles.remove(tile)

        # Spawn new tiles to maintain tiles_ahead count
        furthest_y = max((t.get_y() for t in self._road_tiles), default=0.0)
        while furthest_y < vehicle_y + self.tiles_ahead * self.tile_length:
            furthest_y += self.tile_length
            self._spawn_tile(furthest_y)
            self._next_tile_y = furthest_y + self.tile_length

    # ------------------------------------------------------------------
    # Camera for inference
    # ------------------------------------------------------------------

    def _setup_camera(self) -> None:
        """Set up an offscreen buffer for capturing 224x224 inference frames."""
        from panda3d.core import FrameBufferProperties, WindowProperties, GraphicsPipe
        try:
            fb_props = FrameBufferProperties()
            fb_props.set_rgb_color(True)
            fb_props.set_depth_bits(16)
            win_props = WindowProperties.size(224, 224)
            self._offscreen = self.base.graphics_engine.make_output(
                self.base.pipe,
                "inference_cam",
                -100,
                fb_props,
                win_props,
                GraphicsPipe.BFRefuseWindow | GraphicsPipe.BFSizeTrackHost,
                self.base.win.get_gsg(),
                self.base.win,
            )
            if self._offscreen:
                from panda3d.core import Camera, NodePath
                self._inf_cam = Camera("inf_cam")
                self._inf_cam_np = self.base.render.attach_new_node(self._inf_cam)
                self._inf_cam_np.set_pos(0, -2, 1.2)
                self._inf_cam_np.set_hpr(0, -5, 0)
                dr = self._offscreen.make_display_region()
                dr.set_camera(self._inf_cam_np)
                log.info("Offscreen inference camera created (224x224).")
            else:
                log.warning("Could not create offscreen buffer — using main window screenshot.")
                self._offscreen = None
        except Exception as e:
            log.warning("Offscreen camera setup failed: %s", e)
            self._offscreen = None

    def capture_frame(self) -> np.ndarray:
        """Capture current camera view as (224, 224, 3) uint8 BGR numpy array."""
        try:
            if self._offscreen:
                tex = self.base.graphics_engine.extract_texture_data(
                    self._offscreen.get_texture(), self.base.win.get_gsg()
                )
            # Fallback: screenshot from main window, resize
            from panda3d.core import PNMImage
            img = PNMImage()
            self.base.win.get_screenshot(img)
            w, h = img.get_x_size(), img.get_y_size()
            arr = np.zeros((h, w, 3), dtype=np.uint8)
            for y in range(h):
                for x in range(w):
                    r = int(img.get_red_val(x, y))
                    g = int(img.get_green_val(x, y))
                    b = int(img.get_blue_val(x, y))
                    arr[y, x] = [b, g, r]  # BGR
            import cv2
            return cv2.resize(arr, (224, 224))
        except Exception as e:
            log.warning("Frame capture failed: %s — returning blank frame.", e)
            return np.zeros((224, 224, 3), dtype=np.uint8)

    def attach_vehicle_camera(self, vehicle_node) -> None:
        """Attach the inference camera to the vehicle node."""
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
