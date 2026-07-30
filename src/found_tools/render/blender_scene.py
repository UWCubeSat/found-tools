"""Builds and renders the synthetic Earth scene using Blender's ``bpy`` module.

This is the only module in the render tool that imports ``bpy``.
:func:`found_tools.render.main.main` calls :func:`render_scene` directly, in
process. This module can also be run standalone inside a full Blender
install:

    blender --background --python blender_scene.py -- --scene scene.json --output render.png

where ``scene.json`` is produced by :func:`found_tools.render.scene.build_scene`.
"""

import argparse
import json
import sys
from pathlib import Path

import bpy  # ty: ignore[unresolved-import]
import numpy as np

# WGS84 semi-major (equatorial) and semi-minor (polar) axes. The ~21 km
# difference (flattening ~1/298.257) is small relative to Earth's radius but
# matters for accurate limb-fitting, so the Earth mesh (and the concentric
# cloud/atmosphere shells) are built as WGS84 ellipsoids, not spheres.
WGS84_EQUATORIAL_RADIUS_M = 6_378_137.0
WGS84_POLAR_RADIUS_M = 6_356_752.314245
CLOUD_LAYER_ALTITUDE_M = 10_000.0
# Atmosphere shell radius as a fraction of the Earth's equatorial radius.
# Earth's atmosphere is optically significant to roughly the stratopause
# (~50 km), i.e. under 1% of the Earth's radius; a few percent gives a
# visible glow without the shell reading as a separate, oversized sphere.
ATMOSPHERE_RADIUS_SCALE = 1.02


def _add_wgs84_ellipsoid(
    equatorial_radius: float, segments: int = 256, ring_count: int = 128
):
    """Adds a UV sphere scaled to a WGS84-flattened ellipsoid.

    Blender's primitive_uv_sphere_add only creates true spheres. WGS84 models
    Earth as an oblate spheroid: the polar radius is ~21 km smaller than the
    equatorial radius (flattening ~1/298.257). Build a sphere with the given
    equatorial radius (used for both the X and Y axes), then scale -- and
    bake in -- its local Z axis (the same axis geometry.py's ECI/ECEF frames
    use as Earth's rotation axis) down to the corresponding WGS84 polar
    radius, so cloud/atmosphere shells built at a larger equatorial radius
    stay concentric with, and share the flattening of, the Earth mesh.

    Args:
        equatorial_radius: Equatorial (X/Y) radius of the ellipsoid, meters.
        segments: Longitude subdivisions, passed through to
            primitive_uv_sphere_add.
        ring_count: Latitude subdivisions, passed through to
            primitive_uv_sphere_add.

    Returns:
        The newly created (and active) ellipsoid object.
    """
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=equatorial_radius, segments=segments, ring_count=ring_count
    )
    obj = bpy.context.active_object
    obj.scale.z = WGS84_POLAR_RADIUS_M / WGS84_EQUATORIAL_RADIUS_M
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return obj


def parse_args() -> argparse.Namespace:  # pragma: no cover
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
    """Adds the Earth sphere, textured with Blue Marble and, if the cloud
    texture file exists in texture_dir, a cloud layer."""
    texture_dir = Path(scene["earth"]["texture_dir"])

    earth = _add_wgs84_ellipsoid(WGS84_EQUATORIAL_RADIUS_M)
    earth.name = "Earth"
    # primitive_uv_sphere_add's mesh is flat-shaded by default: every quad
    # face has one constant normal, so despite the 256x128 subdivision the
    # regular lat/long face grid shows up as a faint but very regular
    # checkerboard of brightness steps wherever shading is smooth and
    # low-contrast (e.g. open ocean near the terminator). Smooth-shade so
    # normals are interpolated across faces instead.
    bpy.ops.object.shade_smooth()

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

    cloud_layer_path = texture_dir / scene["earth"]["cloud_layer_filename"]
    if not cloud_layer_path.is_file():
        return

    cloud_radius = WGS84_EQUATORIAL_RADIUS_M + CLOUD_LAYER_ALTITUDE_M
    clouds = _add_wgs84_ellipsoid(cloud_radius)
    clouds.name = "CloudLayer"
    bpy.ops.object.shade_smooth()

    cloud_material = bpy.data.materials.new(name="CloudMaterial")
    cloud_material.use_nodes = True
    cloud_material.blend_method = "BLEND"
    cloud_bsdf = cloud_material.node_tree.nodes["Principled BSDF"]
    cloud_tex = cloud_material.node_tree.nodes.new("ShaderNodeTexImage")
    cloud_tex.image = bpy.data.images.load(str(cloud_layer_path))
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


def add_space_background() -> None:
    """Sets a black deep-space World background.

    An earlier version of this module used Blender's Sky Texture node
    (Nishita / Multiple Scattering) as the World background. That node
    models a ground-level sky dome as seen by an observer standing on the
    Earth's surface -- as a World background it lights the *entire*
    background sphere in every direction (not just near the Earth's limb)
    with its own bright, independent sun disc unrelated to this scene's
    Sun lamp, which is what was blowing out half the frame to white. A
    plain black background is the physically correct choice for a
    from-space view; the limb glow itself is added separately by
    :func:`add_atmosphere_glow`.
    """
    world = bpy.data.worlds.new("SpaceWorld")
    bpy.context.scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes["Background"]
    background.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)


def add_atmosphere_glow(scene: dict) -> None:
    """Adds a sun-facing limb-glow shell around the Earth.

    An earlier version of this glow was driven by view-angle Fresnel alone
    -- which makes every point of the limb glow equally regardless of where
    the Sun actually is, i.e. a uniform white ring even on the shell's night
    side. Real atmospheric limb glow only happens where the atmosphere is
    actually sunlit: it's brightest at the sunlit limb, warms toward orange
    right at the terminator (longer light path through the atmosphere, like
    a sunset on the ground), and is essentially dark on the night side. So
    the glow here is masked by the same Sun vector already used for the Sun
    lamp (dot(surface normal, sun direction)), in addition to the
    view-angle Fresnel term that concentrates it at the limb:

    - Fresnel (grazing view angle) shapes *where on the sphere* the shell
      is visible at all -- the limb, not the disk center.
    - The sun-facing term (surface normal vs. Sun direction) masks that
      down to the sunlit crescent, fading to zero on the night side, and
      drives a colour ramp from warm orange near the terminator to blue on
      the fully sunlit side.
    """
    shell = _add_wgs84_ellipsoid(WGS84_EQUATORIAL_RADIUS_M * ATMOSPHERE_RADIUS_SCALE)
    shell.name = "Atmosphere"
    bpy.ops.object.shade_smooth()

    material = bpy.data.materials.new(name="AtmosphereMaterial")
    material.use_nodes = True
    material.blend_method = "BLEND"
    material.show_transparent_back = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    sun_direction = np.asarray(scene["sun_vector_ecef"], dtype=np.float64)
    sun_direction = sun_direction / np.linalg.norm(sun_direction)

    output = nodes.new("ShaderNodeOutputMaterial")
    mix_shader = nodes.new("ShaderNodeMixShader")
    transparent = nodes.new("ShaderNodeBsdfTransparent")
    emission = nodes.new("ShaderNodeEmission")

    # View-angle term: concentrates the shell's visibility at the limb.
    fresnel = nodes.new("ShaderNodeFresnel")
    fresnel.inputs["IOR"].default_value = 1.2

    # Sun-facing term: dot(surface normal, Sun direction), so the glow is
    # masked to the sunlit side regardless of where the camera is looking
    # from. The Geometry node's Normal output is in world space, which is
    # the same ECEF frame sun_vector_ecef is expressed in, so no extra
    # transform is needed.
    geometry = nodes.new("ShaderNodeNewGeometry")
    sun_dir_node = nodes.new("ShaderNodeCombineXYZ")
    sun_dir_node.inputs["X"].default_value = float(sun_direction[0])
    sun_dir_node.inputs["Y"].default_value = float(sun_direction[1])
    sun_dir_node.inputs["Z"].default_value = float(sun_direction[2])
    sun_dot = nodes.new("ShaderNodeVectorMath")
    sun_dot.operation = "DOT_PRODUCT"
    links.new(geometry.outputs["Normal"], sun_dot.inputs[0])
    links.new(sun_dir_node.outputs["Vector"], sun_dot.inputs[1])

    # Fade the glow in just before the terminator and fully in by a bit
    # past it, instead of a hard day/night cutoff (real atmospheric glow
    # is visible somewhat into twilight).
    day_factor = nodes.new("ShaderNodeMapRange")
    day_factor.clamp = True
    day_factor.inputs["From Min"].default_value = -0.15
    day_factor.inputs["From Max"].default_value = 0.25
    links.new(sun_dot.outputs["Value"], day_factor.inputs["Value"])

    # Colour: warm orange right at the terminator (long path through the
    # atmosphere, like a sunset), shifting to blue further into daylight
    # (shorter, bluer-scattering path), independent of the glow's
    # brightness envelope above.
    color_mix = nodes.new("ShaderNodeMapRange")
    color_mix.clamp = True
    color_mix.inputs["From Min"].default_value = -0.3
    color_mix.inputs["From Max"].default_value = 0.3
    links.new(sun_dot.outputs["Value"], color_mix.inputs["Value"])

    color_ramp = nodes.new("ShaderNodeValToRGB")
    color_ramp.color_ramp.elements[0].position = 0.0
    color_ramp.color_ramp.elements[0].color = (1.0, 0.55, 0.25, 1.0)
    color_ramp.color_ramp.elements[1].position = 1.0
    color_ramp.color_ramp.elements[1].color = (0.35, 0.55, 1.0, 1.0)
    links.new(color_mix.outputs["Result"], color_ramp.inputs["Fac"])
    links.new(color_ramp.outputs["Color"], emission.inputs["Color"])

    glow_factor = nodes.new("ShaderNodeMath")
    glow_factor.operation = "MULTIPLY"
    links.new(fresnel.outputs["Fac"], glow_factor.inputs[0])
    links.new(day_factor.outputs["Result"], glow_factor.inputs[1])

    glow_strength = nodes.new("ShaderNodeMath")
    glow_strength.operation = "MULTIPLY"
    glow_strength.inputs[1].default_value = 6.0
    links.new(glow_factor.outputs["Value"], glow_strength.inputs[0])
    links.new(glow_strength.outputs["Value"], emission.inputs["Strength"])

    links.new(fresnel.outputs["Fac"], mix_shader.inputs["Fac"])
    links.new(transparent.outputs["BSDF"], mix_shader.inputs[1])
    links.new(emission.outputs["Emission"], mix_shader.inputs[2])
    links.new(mix_shader.outputs["Shader"], output.inputs["Surface"])

    shell.data.materials.append(material)


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

    # Blender's default 1000m far-clip plane is far smaller than this
    # scene's ECEF scale (Earth radius alone is ~6.4e6 m), which silently
    # clips the whole Earth out of the render. Give it enough headroom for
    # any orbit up to well past GEO.
    camera_altitude_m = float(np.linalg.norm(cam["position_ecef_m"]))
    cam_data.clip_start = 1.0
    cam_data.clip_end = 4.0 * max(WGS84_EQUATORIAL_RADIUS_M, camera_altitude_m)

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
    # Blender resolves relative paths against the current .blend file, which
    # doesn't exist here, so always pass it an absolute path.
    render_settings.filepath = str(Path(output_path).resolve())
    render_settings.image_settings.file_format = "PNG"
    # AgX (Blender's default) is a filmic tone-mapping curve meant for a
    # cinematic look; it non-linearly recolors pixels, which is undesirable
    # for a tool generating test imagery for attitude-determination software.
    bpy.context.scene.view_settings.view_transform = "Standard"
    # Blender's default of 4096 samples is tuned for offline cinematic
    # rendering and is impractically slow for a CLI tool; this scene has no
    # fine geometric detail (a couple of textured spheres and a sky), so a
    # much lower sample count plus the OIDN denoiser is enough to converge.
    bpy.context.scene.cycles.samples = 128
    bpy.context.scene.cycles.use_denoising = True
    bpy.ops.render.render(write_still=True)


def render_scene(scene: dict, output_path: Path) -> None:
    """Builds and renders a full scene from a scene description dict.

    Args:
        scene: The scene description, as produced by
            :func:`found_tools.render.scene.build_scene`.
        output_path: Path to write the rendered PNG to.
    """
    reset_scene()
    add_earth(scene)
    add_sun(scene)
    add_space_background()
    if scene.get("atmosphere_glow_enabled", True):
        add_atmosphere_glow(scene)
    add_camera(scene)
    render(output_path)


def main() -> None:  # pragma: no cover
    args = parse_args()
    scene = json.loads(Path(args.scene).read_text())
    render_scene(scene, Path(args.output))


if __name__ == "__main__":  # pragma: no cover
    main()
