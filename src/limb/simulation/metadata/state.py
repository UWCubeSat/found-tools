import numpy as np
from scipy.spatial.transform import Rotation as R
import pandas as pd


def generateUniformDirections(num_vectors: int):
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


def positionOnSurfaceEllipsoid(shapeMatrix, direction):
    """
    Calculate the position on the surface of an ellipsoid given a direction vector.

    Parameters:
    diagonalShapeMatrix (3x3 np.ndarray): Diagonal matrix representing the shape of the ellipsoid.
    direction (3d np.ndarray): Direction vector.

    Returns:
    3d np.ndarray: Position on the surface of the ellipsoid.
    """
    # Ensure the direction is a unit vector
    direction = direction / np.linalg.norm(direction)

    # Calculate the scaling factor to reach the surface of the ellipsoid
    scaleFactor = 1 / np.sqrt(direction.T @ shapeMatrix @ direction)

    # Calculate the position on the surface
    position = scaleFactor * direction

    return position

def normSpheroid(shapeMatrix, surfacePoint):
    """
    Calculate the norm of a direction vector with respect to an ellipsoid.

    Parameters:
    shapeMatrix (3x3 np.ndarray): Diagonal matrix representing the shape of the ellipsoid.
    surfacePoint (3d np.ndarray): Point on the surface of the ellipsoid.

    Returns:
    float: Norm of the direction vector with respect to the ellipsoid.
    """
    Sd = shapeMatrix @ surfacePoint
    return Sd / np.linalg.norm(Sd)

def rotateBasis(basis, axis, numRotations):
    thetas = 2 * np.pi * np.arange(numRotations) / numRotations
    rotations = R.from_rotvec(thetas[:, None] * axis).as_matrix()
    return rotations @ basis

def satPositions(shapeMatrix, earthPointDirections, numSatellitePositions, distance):
    surfacePoints = positionOnSurfaceEllipsoid(shapeMatrix, earthPointDirections)
    norm = normSpheroid(shapeMatrix, earthPointDirections)
    normBasis = R.align_vectors([norm], [[0, 0, 1]]).as_matrix()

    length = np.sqrt(distance**2 np.linalg.norm(surfacePoints[0])**2)

    satPositions = np.array([])
    directionToEdge = np.array([])
    for b,s in (normBasis, surfacePoints):
        normBasisVariants = rotateBasis(b, b[:, 2], numSatellitePositions)
        for bv in normBasisVariants:
            directionToEdge.append(-bv[:,0])
            satPositions.append(s + length * bv[:,0])

    return satPositions, directionToEdge

def generateSatelliteState(
    shapeMatrix,
    earthPointDirections,
    distance,
    cameraClass,
    numSatellitePositions,
    numCameraOrientations,
):
    satPositions, satToEdge = satPositions(shapeMatrix, earthPointDirections, distance)

    TCPs = R.align_vectors([satToEdge], [[1, 0, 0]]).as_matrix()

    return satPositions, TCPs


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
