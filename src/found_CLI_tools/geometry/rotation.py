import numpy as np

def gramSchmidt(normal_vector):
    """
    Perform sudo Gram-Schmidt orthogonalization on a set of vectors.

    Parameters:
    normal_vector (3d np.ndarray): single vector to used to create the orthonormal basis.

    Returns:
    matrix of np.ndarray: matrixes of orthonormal vectors with det(M) = 1 (Right handed coordinates system).
    """ 
    l = np.argmax(normal_vector)
    if (l == 0):
        v0 = np.array([normal_vector[1], -normal_vector[0], 0])
    elif (l == 1):
        v0 = np.array([0, normal_vector[2], -normal_vector[1]])
    else:
        v0 = np.array([-normal_vector[2], 0, normal_vector[0]])
    
    v1 = np.cross(normal_vector, v0)

    return np.column_stack((v0 / np.linalg.norm(v0), 
                            v1 / np.linalg.norm(v1),
                            normal_vector / np.linalg.norm(normal_vector)))