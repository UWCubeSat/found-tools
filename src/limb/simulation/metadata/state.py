import numpy as np
from scipy.spatial.transform import Rotation as R


def generate_uniform_directions(num_vectors: int) -> np.ndarray:
    """
    Generates Fibonacci lattice points on a sphere.

    Args:
        num_vectors (int): The number of vectors (points) to generate.

    Returns:
        np.ndarray: Array of Cartesian unit vectors with shape (N, 3),
                    where each row is (x, y, z).
    """
    if num_vectors <= 0:
        return np.empty((0, 3), dtype=np.float64)

    indices = np.arange(num_vectors, dtype=np.float64)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))

    if num_vectors == 1:
        z = np.array([0.0], dtype=np.float64)
    else:
        z = 2.0 * indices / (num_vectors - 1) - 1.0

    azimuth = np.mod(golden_angle * indices, 2.0 * np.pi)
    radial = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    x = radial * np.cos(azimuth)
    y = radial * np.sin(azimuth)

    return np.column_stack((x, y, z))


def _position_on_surface_ellipsoid(shape_matrix: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """
    Calculate the position on the surface of an ellipsoid given a direction vector.

    Parameters:
    shape_matrix (np.ndarray): Ellipsoid shape matrix with shape (3, 3).
    direction (np.ndarray): Direction vector with shape (3,).

    Returns:
    np.ndarray: Position on the surface of the ellipsoid with shape (3,).
    """
    direction = direction / np.linalg.norm(direction)
    scale_factor = 1 / np.sqrt(direction.T @ shape_matrix @ direction)
    return scale_factor * direction

def _norm_spheroid(shape_matrix, surface_point):
    """
    Calculate the norm of a direction vector with respect to an ellipsoid.

    Parameters:
    shape_matrix (np.ndarray): Ellipsoid shape matrix with shape (3, 3).
    surface_point (np.ndarray): Surface point with shape (3,).

    Returns:
    np.ndarray: Unit surface normal with shape (3,).
    """
    sd = shape_matrix @ surface_point
    return sd / np.linalg.norm(sd)

def _rotate_basis(basis, axis, num_rotations):
    thetas = 2 * np.pi * np.arange(num_rotations) / num_rotations
    rotations = R.from_rotvec(thetas[:, None] * axis).as_matrix()
    return rotations @ basis

def _sat_position_boresight(shape_matrix, earth_point_directions, num_satellite_positions, distance):
    """
    Generate satellite positions around points on an ellipsoid.

    Parameters:
    shape_matrix (np.ndarray): Ellipsoid shape matrix with shape (3, 3).
    earth_point_directions (np.ndarray): Unit direction vectors with shape (N, 3).
    num_satellite_positions (int): Number of satellite positions per earth point.
    distance (float): Distance from the origin to the satellite.

    Returns:
    tuple[np.ndarray, rotation]:
        - sat_positions with shape (N * num_satellite_positions, 3)
        - direction_to_edge with shape (N * num_satellite_positions, 3)
    """
    assert np.isclose(np.linalg.norm(earth_point_directions, axis=1), 1.0).all(), (
        "earth_point_directions must be unit vectors"
    )

    sat_positions = []
    sat_to_edge = []
    for direction in earth_point_directions:
        surface_point = _position_on_surface_ellipsoid(shape_matrix, direction)
        norm = _norm_spheroid(shape_matrix, direction)
        norm_basis = R.align_vectors([norm], [[0, 0, 1]])[0].as_matrix()
        length = np.sqrt(distance**2 - np.linalg.norm(surface_point) ** 2)
        norm_basis_variants = _rotate_basis(norm_basis, norm_basis[:, 2], num_satellite_positions)
        for bv in norm_basis_variants:
            sat_to_edge.append(-bv[:, 0])
            sat_positions.append(surface_point + length * bv[:, 0])

    sat_positions = np.array(sat_positions)
    sat_to_edge = np.array(sat_to_edge)
    tcps = R.align_vectors(sat_to_edge, np.tile([1, 0, 0], (len(sat_to_edge), 1)))[0]

    return sat_positions, tcps


# def generateSatelliteState(
#     shapeMatrix,
#     earthPointDirection,
#     distance,
#     cameraClass,
#     numSatellitePositions,
#     numSatelliteOrientations,
# ):
#     """Generate satellite positions and orientations around an ellipsoid.

#     Creates a grid of satellite positions at fixed distance from a target ellipsoid,
#     arranged in a circle around the tangent plane at earthPointDirection. For each
#     position, generates multiple camera orientations by rotating around the boresight.

#     Args:
#         shapeMatrix: 3x3 symmetric matrix defining the ellipsoid (diagonal elements
#                      are 1/a² for semi-axes a, b, c).
#         earthPointDirection: Unit vector (shape 3) pointing to target location on ellipsoid.
#         distance: Satellite distance from origin. Must exceed the maximum semi-axis;
#                   satellites are always outside the ellipsoid body.
#         cameraClass: Camera object with focal length, resolution, and edge-offset properties.
#         numSatellitePositions: Number of positions around the tangent plane.
#         numSatelliteOrientations: Number of orientations per position (boresight angles).

#     Returns:
#         Tuple[np.ndarray, np.ndarray]: Satellite positions (shape numSatellitePositions, 3)
#         and orientations (shape numSatelliteOrientations × numSatellitePositions, 3, 3).

#     Raises:
#         AssertionError: If distance <= max semi-axis of the ellipsoid.
#     """
#     # Preconditions: earthPointDirection is a unit vector; distance > max(semi-axes).
#     max_semi_axis = 1.0 / np.sqrt(np.min(np.diag(shapeMatrix)))
#     assert distance > max_semi_axis, (
#         f"Satellite distance {distance} must be > max semi-axis {max_semi_axis}"
#     )
#     assert np.isclose(np.linalg.norm(earthPointDirection), 1.0), (
#         "earthPointDirection must be a unit vector"
#     )

#     # noraml direction fomr ellipsoid is not same as direction to the edge since it is a spheroid.
#     Sd = shapeMatrix @ earthPointDirection
#     normaldirection = Sd / np.linalg.norm(Sd)
#     # Create orthonormal basis: earthPointDirection is the Z-axis;
#     # X and Y are tangent to the ellipsoid surface at this direction.
#     tangentBasis = R.align_vectors([normaldirection], [[0, 0, 1]])[0].as_matrix()

#     # Generate numSatellitePositions around the tangent plane.
#     thetas = 2 * np.pi * np.arange(numSatellitePositions) / numSatellitePositions
#     satPositionRotations = R.from_rotvec(thetas[:, None] * tangentBasis[:, 2]).as_matrix()
#     rotatedTangentBasis = satPositionRotations @ tangentBasis

#     # call now so that all orientations use same pixel edge angle
#     imageEdgeAngle = cameraClass.edgeAngle()

#     # Rotation sequence: flip E0 toward edge → rotate away from edge → rotate around boresight.
#     rotationFlipE0 = R.from_rotvec(np.pi * rotatedTangentBasis[0, 2]).as_matrix()
#     rotationOffImagePixel = R.from_rotvec(
#         imageEdgeAngle * rotatedTangentBasis[:, 1]
#     ).as_matrix()

#     phi = 2 * np.pi * np.arange(numSatelliteOrientations) / numSatelliteOrientations
#     boresight_rotvecs = phi[:, None, None] * rotatedTangentBasis[None, :, 0, :]
#     # Reshape from (numSatelliteOrientations, numSatellitePositions, 3) to (M*N, 3)
#     # for scipy's Rotation.from_rotvec, then reshape back to (M, N, 3, 3).
#     boresight_rotvecs_flat = boresight_rotvecs.reshape(-1, 3)
#     rotations_flat = R.from_rotvec(boresight_rotvecs_flat).as_matrix()
#     rotationAroundBoresight = rotations_flat.reshape(
#         numSatelliteOrientations, numSatellitePositions, 3, 3
#     )

#     # Composite orientation: shape (numSatelliteOrientations, numSatellitePositions, 3, 3).
#     satelliteOrientations = (
#         rotationAroundBoresight
#         @ rotationOffImagePixel[None, ...]
#         @ rotationFlipE0[None, ...]
#         @ rotatedTangentBasis[None, ...]
#     )

#     # Compute satellite positions at each tangent-plane location.
#     # Use the first column of the first orientation's basis as the radial direction.
#     surfacePoint = positionOnSurfaceEllipsoid(shapeMatrix, earthPointDirection)
#     distanceFromEdgePoint = np.sqrt(distance**2 - np.linalg.norm(surfacePoint) ** 2)
#     radialDirection = rotatedTangentBasis[:, :, 0]
#     satboresight = satelliteOrientations[0, :, :, 0]
#     satellitePositions = (
#         surfacePoint + distanceFromEdgePoint * radialDirection
#     )

#     return satellitePositions, satelliteOrientations
