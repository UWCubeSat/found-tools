import random
import numpy as np
from scipy.spatial.transform import Rotation as R


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
    position = np.sqrt(scaleFactor) * direction

    return position


def generateSatelliteState(
    shapeMatrix,
    earthPointDirection,
    distance,
    cameraClass,
    edgeOffset,
    numSatellitePositions,
    numSatelliteOrientations,
):
    """Generate satellite positions and orientations around an ellipsoid.

    Creates a grid of satellite positions at fixed distance from a target ellipsoid,
    arranged in a circle around the tangent plane at earthPointDirection. For each
    position, generates multiple camera orientations by rotating around the boresight.

    Args:
        shapeMatrix: 3x3 symmetric matrix defining the ellipsoid (diagonal elements
                     are 1/a² for semi-axes a, b, c).
        earthPointDirection: Unit vector (shape 3) pointing to target location on ellipsoid.
        distance: Satellite distance from origin. Must exceed the maximum semi-axis;
                  satellites are always outside the ellipsoid body.
        cameraClass: Camera object with focal length and resolution properties.
        edgeOffset: Pixel margin from image boundary used to constrain edge angle.
        numSatellitePositions: Number of positions around the tangent plane.
        numSatelliteOrientations: Number of orientations per position (boresight angles).

    Returns:
        Tuple[np.ndarray, np.ndarray]: Satellite positions (shape numSatellitePositions, 3)
        and orientations (shape numSatelliteOrientations × numSatellitePositions, 3, 3).

    Raises:
        AssertionError: If distance <= max semi-axis of the ellipsoid.
    """
    # Preconditions: earthPointDirection is a unit vector; distance > max(semi-axes).
    max_semi_axis = 1.0 / np.sqrt(np.min(np.diag(shapeMatrix)))
    assert distance > max_semi_axis, (
        f"Satellite distance {distance} must be > max semi-axis {max_semi_axis}"
    )

    # Create orthonormal basis: earthPointDirection is the Z-axis;
    # X and Y are tangent to the ellipsoid surface at this direction.
    tangentBasis = R.align_vectors([earthPointDirection], [[0, 0, 1]])[0].as_matrix()

    # Generate numSatellitePositions around the tangent plane.
    thetas = 2 * np.pi * np.arange(numSatellitePositions) / numSatellitePositions
    rotations = R.from_rotvec(thetas[:, None] * tangentBasis[:, 2]).as_matrix()
    rotatedTangentBasis = rotations @ tangentBasis

    # Compute edge angle for camera: constrained by image bounds minus edge offset.
    smallestImageDimension = np.min(
        [
            cameraClass.xResolution_ * cameraClass.xPixelPitch_,
            cameraClass.yResolution_ * cameraClass.yPixelPitch_,
        ]
    )
    imagePoint = np.array(
        [cameraClass.focalLength_, (smallestImageDimension - edgeOffset) / 2, 0]
    )
    imageMaxEdgeAngle = np.arccos(
        np.dot(np.array([1, 0, 0]), imagePoint) / np.linalg.norm(imagePoint)
    )
    imageEdgeAngle = random.uniform(0, imageMaxEdgeAngle)

    # Rotation sequence: flip E0 toward edge → rotate away from edge → rotate around boresight.
    rotationFlipE0 = R.from_rotvec(np.pi * rotatedTangentBasis[:, 2]).as_matrix()
    rotationOffImagePixel = R.from_rotvec(
        imageEdgeAngle * rotatedTangentBasis[:, 1]
    ).as_matrix()
    phi = 2 * np.pi * np.arange(numSatelliteOrientations) / numSatelliteOrientations
    boresight_rotvecs = phi[:, None, None] * rotatedTangentBasis[None, :, 0, :]
    # Reshape from (numSatelliteOrientations, numSatellitePositions, 3) to (M*N, 3)
    # for scipy's Rotation.from_rotvec, then reshape back to (M, N, 3, 3).
    boresight_rotvecs_flat = boresight_rotvecs.reshape(-1, 3)
    rotations_flat = R.from_rotvec(boresight_rotvecs_flat).as_matrix()
    rotationAroundBoresight = rotations_flat.reshape(
        numSatelliteOrientations, numSatellitePositions, 3, 3
    )

    # Composite orientation: shape (numSatelliteOrientations, numSatellitePositions, 3, 3).
    satelliteOrientations = (
        rotationAroundBoresight
        @ rotationOffImagePixel[None, ...]
        @ rotationFlipE0[None, ...]
        @ rotatedTangentBasis[None, ...]
    )

    # Compute satellite positions at each tangent-plane location.
    surfacePoint = positionOnSurfaceEllipsoid(shapeMatrix, earthPointDirection)
    distanceFromEdgePoint = np.sqrt(distance**2 - np.linalg.norm(surfacePoint) ** 2)
    satellitePositions = (
        surfacePoint + distanceFromEdgePoint * rotatedTangentBasis[:, 0]
    )

    return satellitePositions, satelliteOrientations
