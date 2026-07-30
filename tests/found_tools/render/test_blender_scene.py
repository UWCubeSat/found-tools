from pathlib import Path

import bpy
import pytest
from PIL import Image

from found_tools.render import blender_scene
from found_tools.render.blender_scene import (
    WGS84_EQUATORIAL_RADIUS_M,
    WGS84_POLAR_RADIUS_M,
    add_atmosphere_glow,
    add_camera,
    add_earth,
    add_space_background,
    add_sun,
    render,
    render_scene,
    reset_scene,
)


@pytest.fixture
def scene(tmp_path: Path) -> dict:
    blue_marble = "world.202603.3x5400x2700.jpg"
    cloud_layer = "cloud_combined_2048.jpg"
    Image.new("RGB", (2, 2)).save(tmp_path / blue_marble)
    Image.new("RGB", (2, 2)).save(tmp_path / cloud_layer)

    return {
        "date": "2026-03-20T12:00:00+00:00",
        "sun_vector_ecef": [1.0, 0.0, 0.0],
        "camera": {
            "focal_length_m": 0.05,
            "x_pixel_pitch_m": 5e-6,
            "y_pixel_pitch_m": 5e-6,
            "x_resolution": 1920,
            "y_resolution": 1080,
            "x_center": 960.0,
            "y_center": 540.0,
            "position_ecef_m": [7_000_000.0, 0.0, 0.0],
            "rotation_ecef_to_camera": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        },
        "earth": {
            "blue_marble_filename": blue_marble,
            "cloud_layer_filename": cloud_layer,
            "texture_dir": str(tmp_path),
        },
        "image": {"width": 1920, "height": 1080},
    }


@pytest.fixture(autouse=True)
def _reset_blender_scene():
    reset_scene()
    yield
    reset_scene()


def test_reset_scene_clears_existing_objects():
    bpy.ops.mesh.primitive_cube_add()

    reset_scene()

    assert len(bpy.data.objects) == 0


def test_add_earth_creates_textured_earth_and_cloud_layer(scene):
    add_earth(scene)

    assert "Earth" in bpy.data.objects
    assert "CloudLayer" in bpy.data.objects
    earth_material = bpy.data.objects["Earth"].data.materials[0]
    assert earth_material.name == "EarthMaterial"
    cloud_material = bpy.data.objects["CloudLayer"].data.materials[0]
    assert cloud_material.name == "CloudMaterial"
    assert cloud_material.blend_method == "BLEND"
    # Flat-shaded quads on the sphere's regular lat/long face grid produce a
    # visible checkerboard of brightness steps under smooth, low-contrast
    # lighting (e.g. open ocean near the terminator); every face must be
    # smooth-shaded so normals interpolate across the sphere instead.
    assert all(
        polygon.use_smooth for polygon in bpy.data.objects["Earth"].data.polygons
    )
    assert all(
        polygon.use_smooth for polygon in bpy.data.objects["CloudLayer"].data.polygons
    )


def _bounding_radii(obj) -> tuple[float, float]:
    """Returns (max equatorial radius, polar radius) of a mesh's vertices."""
    coords = [obj.matrix_world @ v.co for v in obj.data.vertices]
    equatorial = max((c.x**2 + c.y**2) ** 0.5 for c in coords)
    polar = max(abs(c.z) for c in coords)
    return equatorial, polar


def test_add_earth_is_a_wgs84_ellipsoid_not_a_sphere(scene):
    # A sphere would have equatorial radius == polar radius. WGS84 models
    # Earth as an oblate spheroid: ~21 km flatter at the poles than at the
    # equator, which matters for accurate limb-fitting.
    add_earth(scene)

    equatorial, polar = _bounding_radii(bpy.data.objects["Earth"])
    assert equatorial == pytest.approx(WGS84_EQUATORIAL_RADIUS_M, rel=1e-3)
    assert polar == pytest.approx(WGS84_POLAR_RADIUS_M, rel=1e-3)
    assert equatorial - polar == pytest.approx(21_384.685755, abs=100.0)


def test_add_earth_cloud_layer_shares_earths_flattening(scene):
    add_earth(scene)

    equatorial, polar = _bounding_radii(bpy.data.objects["CloudLayer"])
    expected_ratio = WGS84_POLAR_RADIUS_M / WGS84_EQUATORIAL_RADIUS_M
    assert polar / equatorial == pytest.approx(expected_ratio, rel=1e-6)


def test_add_earth_skips_cloud_layer_when_texture_missing(scene, tmp_path):
    (tmp_path / scene["earth"]["cloud_layer_filename"]).unlink()

    add_earth(scene)

    assert "Earth" in bpy.data.objects
    assert "CloudLayer" not in bpy.data.objects


def test_add_sun_creates_sun_light_pointing_away_from_sun_vector(scene):
    add_sun(scene)

    assert "Sun" in bpy.data.objects
    sun = bpy.data.objects["Sun"]
    assert sun.data.energy == 3.0
    assert sun.rotation_mode == "QUATERNION"
    assert len(sun.rotation_quaternion) == 4


def test_add_space_background_sets_black_background():
    add_space_background()

    world = bpy.context.scene.world
    assert world is not None
    background = world.node_tree.nodes["Background"]
    assert tuple(background.inputs["Color"].default_value) == (0.0, 0.0, 0.0, 1.0)


def test_add_atmosphere_glow_creates_shell(scene):
    add_atmosphere_glow(scene)

    assert "Atmosphere" in bpy.data.objects
    shell = bpy.data.objects["Atmosphere"]
    material = shell.data.materials[0]
    node_types = {node.bl_idname for node in material.node_tree.nodes}
    assert "ShaderNodeFresnel" in node_types
    assert all(polygon.use_smooth for polygon in shell.data.polygons)
    assert "ShaderNodeEmission" in node_types
    assert "ShaderNodeBsdfTransparent" in node_types


def test_add_atmosphere_glow_shell_shares_earths_flattening(scene):
    add_atmosphere_glow(scene)

    equatorial, polar = _bounding_radii(bpy.data.objects["Atmosphere"])
    expected_ratio = WGS84_POLAR_RADIUS_M / WGS84_EQUATORIAL_RADIUS_M
    assert polar / equatorial == pytest.approx(expected_ratio, rel=1e-6)


def test_add_atmosphere_glow_is_masked_by_sun_direction(scene):
    # The glow's brightness/colour must depend on the same Sun direction
    # used for the Sun lamp, not view-angle Fresnel alone -- otherwise the
    # shell renders as a uniform white ring regardless of where the Sun is.
    add_atmosphere_glow(scene)

    material = bpy.data.objects["Atmosphere"].data.materials[0]
    nodes = material.node_tree.nodes
    node_types = {node.bl_idname for node in nodes}
    assert "ShaderNodeNewGeometry" in node_types
    assert "ShaderNodeVectorMath" in node_types
    assert "ShaderNodeValToRGB" in node_types

    sun_dir_node = next(n for n in nodes if n.bl_idname == "ShaderNodeCombineXYZ")
    assert (
        sun_dir_node.inputs["X"].default_value,
        sun_dir_node.inputs["Y"].default_value,
        sun_dir_node.inputs["Z"].default_value,
    ) == pytest.approx(tuple(scene["sun_vector_ecef"]))

    dot_node = next(
        n
        for n in nodes
        if n.bl_idname == "ShaderNodeVectorMath" and n.operation == "DOT_PRODUCT"
    )
    geometry_node = next(n for n in nodes if n.bl_idname == "ShaderNodeNewGeometry")
    assert dot_node.inputs[0].links[0].from_node == geometry_node
    assert dot_node.inputs[1].links[0].from_node == sun_dir_node

    emission = next(n for n in nodes if n.bl_idname == "ShaderNodeEmission")
    # Strength must be driven by the shader graph (Fresnel * sun-facing
    # factor), not left at a flat, direction-independent constant.
    assert emission.inputs["Strength"].links
    color_ramp = next(n for n in nodes if n.bl_idname == "ShaderNodeValToRGB")
    assert emission.inputs["Color"].links[0].from_node == color_ramp
    # Warm near the terminator, blue further into daylight.
    assert tuple(color_ramp.color_ramp.elements[0].color) == pytest.approx(
        (1.0, 0.55, 0.25, 1.0)
    )
    assert tuple(color_ramp.color_ramp.elements[1].color) == pytest.approx(
        (0.35, 0.55, 1.0, 1.0)
    )


def test_add_atmosphere_glow_normalizes_sun_vector(scene):
    scene["sun_vector_ecef"] = [2.0, 0.0, 0.0]

    add_atmosphere_glow(scene)

    material = bpy.data.objects["Atmosphere"].data.materials[0]
    sun_dir_node = next(
        n for n in material.node_tree.nodes if n.bl_idname == "ShaderNodeCombineXYZ"
    )
    assert sun_dir_node.inputs["X"].default_value == pytest.approx(1.0)


def test_add_camera_sets_intrinsics_resolution_and_pose(scene):
    add_camera(scene)

    camera = bpy.context.scene.camera
    assert camera is not None
    assert camera.data.lens == pytest.approx(scene["camera"]["focal_length_m"] * 1000.0)
    assert camera.data.sensor_fit == "HORIZONTAL"
    assert bpy.context.scene.render.resolution_x == scene["image"]["width"]
    assert bpy.context.scene.render.resolution_y == scene["image"]["height"]
    assert list(camera.location) == pytest.approx(scene["camera"]["position_ecef_m"])
    # Blender's default 1000m far-clip plane is far smaller than this ECEF
    # scene, and would silently clip the Earth sphere out of the render.
    assert camera.data.clip_end > 7_000_000.0


def test_render_configures_settings_and_writes_output(tmp_path):
    # A real (but tiny) render, exercising the actual bpy.ops.render.render
    # call rather than mocking it away.
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0)
    bpy.ops.object.light_add(type="SUN")
    cam_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    camera.location = (0.0, 0.0, 5.0)
    bpy.context.scene.render.resolution_x = 4
    bpy.context.scene.render.resolution_y = 4
    output_path = tmp_path / "render.png"

    render(output_path)

    render_settings = bpy.context.scene.render
    assert render_settings.engine == "CYCLES"
    assert render_settings.filepath == str(output_path)
    assert render_settings.image_settings.file_format == "PNG"
    assert bpy.context.scene.view_settings.view_transform == "Standard"
    assert bpy.context.scene.cycles.samples == 128
    assert bpy.context.scene.cycles.use_denoising is True
    assert output_path.is_file()


def test_render_resolves_relative_output_path(tmp_path, monkeypatch):
    # Blender treats a relative render filepath as relative to the current
    # .blend file (which doesn't exist here), so a relative output_path must
    # still resolve against the working directory rather than erroring.
    monkeypatch.chdir(tmp_path)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0)
    bpy.ops.object.light_add(type="SUN")
    cam_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    camera.location = (0.0, 0.0, 5.0)
    bpy.context.scene.render.resolution_x = 4
    bpy.context.scene.render.resolution_y = 4
    bpy.context.scene.cycles.samples = 1

    render(Path("render.png"))

    assert (tmp_path / "render.png").is_file()


def test_render_scene_builds_scene_and_calls_render(monkeypatch, scene, tmp_path):
    calls = []
    monkeypatch.setattr(
        blender_scene, "render", lambda output_path: calls.append(output_path)
    )
    output_path = tmp_path / "render.png"

    render_scene(scene, output_path)

    assert "Earth" in bpy.data.objects
    assert "Sun" in bpy.data.objects
    assert "Atmosphere" in bpy.data.objects
    assert bpy.context.scene.camera is not None
    assert bpy.context.scene.world is not None
    assert calls == [output_path]


def test_render_scene_skips_atmosphere_glow_shell_when_disabled(
    monkeypatch, scene, tmp_path
):
    monkeypatch.setattr(blender_scene, "render", lambda output_path: None)
    scene["atmosphere_glow_enabled"] = False

    render_scene(scene, tmp_path / "render.png")

    assert "Earth" in bpy.data.objects
    assert "Atmosphere" not in bpy.data.objects
    # Disabling the glow shell shouldn't affect the black deep-space
    # background.
    world = bpy.context.scene.world
    assert world is not None
    background = world.node_tree.nodes["Background"]
    assert tuple(background.inputs["Color"].default_value) == (0.0, 0.0, 0.0, 1.0)


def test_render_scene_adds_atmosphere_glow_shell_by_default(monkeypatch, scene):
    # scene fixtures built via found_tools.render.scene.build_scene always
    # set this key, but blender_scene.render_scene should default to
    # enabling the glow shell even if a hand-built scene dict omits it.
    monkeypatch.setattr(blender_scene, "render", lambda output_path: None)
    scene.pop("atmosphere_glow_enabled", None)

    render_scene(scene, Path("render.png"))

    assert "Atmosphere" in bpy.data.objects
