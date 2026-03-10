from random import random
import numpy as np
from scipy.spatial.transform import Rotation as R


def generateUniformDirections():
    return np.random.normal(size=(3, 3))


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
    # @pre earthPointDirection is a unit vector & distance > the max axis of the ellipsoid defined
    # by shapeMatrix
    # @note assumes planet is centered at the origin
    # it is interesting that we can find the orientations of the satellites first without the position of
    # the satellite if we know some properties about the camera and where we want the edge to appear in our image

    ### find the orientations of the satellite

    # create rotation matrix that brings the z axis to earthPointDirection x and y axes
    # are now tangent to the surface of the ellipsoid at the point defined by earthPointDirection
    tangentBasis = R.align_vectors([earthPointDirection], [[0, 0, 1]])[0].as_matrix()

    # initialize satellite positions in the tangent plane
    thetas = 2 * np.pi * np.arange(numSatellitePositions) / numSatellitePositions
    rotations = R.from_rotvec(thetas[:, None] * tangentBasis[:, 2]).as_matrix()
    rotatedTangentBasis = rotations @ tangentBasis

    # rotatedTangentBasis has shape (numSatellitePositions, 3, 3) where the last dimension is the
    # orthonormal basis vectors E0, E1, E2

    # edge pixel of choice should fall within the circle inscribed by the rectangle of the image plane
    # minus the edge offset
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

    # rotation around the E2 basis vector (surface normal) to point the E0 basis vector towards the
    # edge pixel coordinate when the satellite is at its postion in the tangent plane
    rotationFlipE0 = R.from_rotvec(np.pi * rotatedTangentBasis[:, 2]).as_matrix()
    # rotation away from edge pixel coordinate
    rotationOffImagePixel = R.from_rotvec(
        imageEdgeAngle * rotatedTangentBasis[:, 1]
    ).as_matrix()
    # rotation around the camera boresight (E0 basis vector)
    phi = 2 * np.pi * np.arange(numSatelliteOrientations) / numSatelliteOrientations
    rotationAroundBoresight = R.from_rotvec(
        phi[:, None] * rotatedTangentBasis[:, 0]
    ).as_matrix()

    # find orientation of the satellite
    # rotationFromTangentBasis has shape (numSatelliteOrientations, numSatellitePositions, 3, 3)
    # where the last dimension is the rotation matrix from the tangent basis to the satellite orientation
    satelliteOrientations = (
        rotationAroundBoresight
        @ rotationOffImagePixel
        @ rotationFlipE0
        @ rotatedTangentBasis
    )

    ### find the position of the satellite

    # find vector to point on surface of ellipsoid
    surfacePoint = positionOnSurfaceEllipsoid(shapeMatrix, earthPointDirection)
    # find distance from surface point to satellite position
    distanceFromEdgePoint = np.sqrt(distance**2 - np.linalg.norm(surfacePoint) ** 2)
    # find satellite positions in equatorial coordinates
    satellitePositions = (
        surfacePoint + distanceFromEdgePoint * rotatedTangentBasis[:, 0]
    )

    return satellitePositions, satelliteOrientations
