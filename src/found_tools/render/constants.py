# Time Constants
J2000_JD = 2451545.0  # Julian Date of the J2000.0 epoch (2000-01-01T12:00:00 UTC)
DAYS_PER_JULIAN_CENTURY = 36525.0

# Earth Constants
EARTH_OBLIQUITY_DEG = 23.439  # Mean obliquity of the ecliptic at J2000, degrees
EARTH_OBLIQUITY_RATE_DEG_PER_DAY = -0.0000004  # Secular drift of the obliquity

# Blue Marble Next Generation texture set: one true-color mosaic per month,
# chosen so the rendered scene's cloud/vegetation state roughly matches the
# requested date. Filenames follow NASA's published naming convention.
BLUE_MARBLE_MONTH_FILENAMES = {
    1: "world.200401.3x5400x2700.jpg",
    2: "world.200402.3x5400x2700.jpg",
    3: "world.200403.3x5400x2700.jpg",
    4: "world.200404.3x5400x2700.jpg",
    5: "world.200405.3x5400x2700.jpg",
    6: "world.200406.3x5400x2700.jpg",
    7: "world.200407.3x5400x2700.jpg",
    8: "world.200408.3x5400x2700.jpg",
    9: "world.200409.3x5400x2700.jpg",
    10: "world.200410.3x5400x2700.jpg",
    11: "world.200411.3x5400x2700.jpg",
    12: "world.200412.3x5400x2700.jpg",
}

# Generic cloud layer texture, reused for every month (see caveat in the tool
# README: NASA does not publish a month-by-month cloud product).
CLOUD_LAYER_FILENAME = "cloud_combined_2048.jpg"

DEFAULT_IMAGE_WIDTH = 1920
DEFAULT_IMAGE_HEIGHT = 1080

# Default output scene description filename, consumed by blender_scene.py
DEFAULT_SCENE_FILENAME = "scene.json"
