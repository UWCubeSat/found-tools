"""Builds and renders the synthetic Earth scene inside Blender.

This is the only module in the render tool that imports ``bpy``. It is not
part of the ``found_tools`` package's normal import graph and is not
exercised by the test suite (``bpy`` is Blender's bundled interpreter
module and is not installable for the project's supported Python version --
see the render tool README for how to run this script). It is invoked as:

    blender --background --python blender_scene.py -- --scene scene.json --output render.png

where ``scene.json`` is produced by :func:`found_tools.render.scene.build_scene`.
"""

import argparse
import json
import sys
from pathlib import Path

import bpy  # ty: ignore[unresolved-import]

EARTH_RADIUS_M = 6_378_137.0
CLOUD_LAYER_ALTITUDE_M = 10_000.0


def parse_args() -> argparse.Namespace:
    """Parses the arguments passed after Blender's own ``--`` separator."""
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []

    parser = argparse.ArgumentParser(
        description="Render a synthetic Earth-limb image from a scene description."
    )
    parser.add_argument("--scene", required=True, help="Path to the scene JSON file.")
    parser.add_argument(
        "--output", required=True, help="Path to write the rendered PNG to."
    )
    return parser.parse_args(argv)


def reset_scene() -> None:
    """Clears the default Blender scene."""
    bpy.ops.wm.read_factory_settings(use_empty=True)


def add_earth(scene: dict) -> None:
    """Adds the Earth sphere, textured with Blue Marble and a cloud layer."""
    texture_dir = Path(scene["earth"]["texture_dir"])

    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=EARTH_RADIUS_M, segments=256, ring_count=128
    )
    earth = bpy.context.active_object
    earth.name = "Earth"

    earth_material = bpy.data.materials.new(name="EarthMaterial")
    earth_material.use_nodes = True
    bsdf = earth_material.node_tree.nodes["Principled BSDF"]
    tex_image = earth_material.node_tree.nodes.new("ShaderNodeTexImage")
    tex_image.image = bpy.data.images.load(
        str(texture_dir / scene["earth"]["blue_marble_filename"])
    )
    earth_material.node_tree.links.new(
        bsdf.inputs["Base Color"], tex_image.outputs["Color"]
    )
    earth.data.materials.append(earth_material)

    cloud_radius = EARTH_RADIUS_M + CLOUD_LAYER_ALTITUDE_M
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=cloud_radius, segments=256, ring_count=128
    )
    clouds = bpy.context.active_object
    clouds.name = "CloudLayer"

    cloud_material = bpy.data.materials.new(name="CloudMaterial")
    cloud_material.use_nodes = True
    cloud_material.blend_method = "BLEND"
    cloud_bsdf = cloud_material.node_tree.nodes["Principled BSDF"]
    cloud_tex = cloud_material.node_tree.nodes.new("ShaderNodeTexImage")
    cloud_tex.image = bpy.data.images.load(
        str(texture_dir / scene["earth"]["cloud_layer_filename"])
    )
    cloud_material.node_tree.links.new(
        cloud_bsdf.inputs["Alpha"], cloud_tex.outputs["Color"]
    )
    cloud_material.node_tree.links.new(
        cloud_bsdf.inputs["Base Color"], cloud_tex.outputs["Color"]
    )
    clouds.data.materials.append(cloud_material)


def add_sun(scene: dict) -> None:
    """Adds a Sun lamp pointed along the scene's Sun vector."""
    sun_vector = scene["sun_vector_ecef"]

    bpy.ops.object.light_add(type="SUN")
    sun = bpy.context.active_object
    sun.name = "Sun"
    sun.data.energy = 3.0

    # Blender's default Sun lamp shines along -Z in its local frame; point
    # -Z at the direction *from* the Sun *to* Earth (i.e. along -sun_vector).
    direction = [-c for c in sun_vector]
    sun.rotation_mode = "QUATERNION"
    sun.rotation_quaternion = (
        __import__("mathutils")
        .Vector((0, 0, -1))
        .rotation_difference(__import__("mathutils").Vector(direction))
    )


def add_atmosphere() -> None:
    """Enables Blender's Nishita sky texture for limb glow / terminator softening."""
    world = bpy.data.worlds.new("AtmosphereWorld")
    bpy.context.scene.world = world
    world.use_nodes = True
    sky_texture = world.node_tree.nodes.new("ShaderNodeTexSky")
    sky_texture.sky_type = "NISHITA"
    background = world.node_tree.nodes["Background"]
    world.node_tree.links.new(background.inputs["Color"], sky_texture.outputs["Color"])


def add_camera(scene: dict) -> None:
    """Adds a camera whose intrinsics and pose match the scene description."""
    cam_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(camera)
    bpy.context.scene.camera = camera

    cam = scene["camera"]

    cam_data.lens = cam["focal_length_m"] * 1000.0  # Blender lens is in mm
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.sensor_width = cam["x_resolution"] * cam["x_pixel_pitch_m"] * 1000.0
    cam_data.sensor_height = cam["y_resolution"] * cam["y_pixel_pitch_m"] * 1000.0

    render = bpy.context.scene.render
    render.resolution_x = scene["image"]["width"]
    render.resolution_y = scene["image"]["height"]

    camera.location = cam["position_ecef_m"]

    # Blender cameras look down local -Z with +Y up; the pinhole model here
    # (see found_tools.utils.Camera) looks down local +X, so rotate that
    # frame into Blender's before applying the ECEF->camera rotation.
    mathutils = __import__("mathutils")
    pinhole_to_blender = mathutils.Matrix(((0, 0, -1), (-1, 0, 0), (0, 1, 0)))
    ecef_to_camera = mathutils.Matrix(cam["rotation_ecef_to_camera"])
    camera_to_ecef = ecef_to_camera.transposed()
    camera.matrix_world = (
        mathutils.Matrix.Translation(camera.location)
        @ (camera_to_ecef @ pinhole_to_blender).to_4x4()
    )


def render(output_path: Path) -> None:
    """Configures render settings and renders the scene to ``output_path``."""
    render_settings = bpy.context.scene.render
    render_settings.engine = "CYCLES"
    render_settings.filepath = str(output_path)
    render_settings.image_settings.file_format = "PNG"
    bpy.ops.render.render(write_still=True)


def main() -> None:
    args = parse_args()
    scene = json.loads(Path(args.scene).read_text())

    reset_scene()
    add_earth(scene)
    add_sun(scene)
    add_atmosphere()
    add_camera(scene)
    render(Path(args.output))


if __name__ == "__main__":
    main()
