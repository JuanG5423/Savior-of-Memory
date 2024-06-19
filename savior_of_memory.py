import numpy as np
from scipy import linalg as la

def svd(A: np.array, error_tolerance: float=10**-50):
    #Get the eigenvalues and eigenvectors of (A^T)A
    eigenvalues, eigenvectors = np.linalg.eig(np.dot(A.conj().T, A))
    
    #Calculate the singular values
    singular_values = eigenvalues ** 0.5

    #Count the number of nonzero singular values
    rank = len([singular_value for singular_value in singular_values if singular_value > error_tolerance])

    #Keep only the positive singular values
    positive_singular_values = np.array([singular_values[i] for i in range(rank)])

    #Keep only the corresponding eigenvectors
    positive_eigenvectors = np.array([eigenvectors[i] for i in range(rank)])

    #TODO: Construct U with array broadcasting

    return singular_values, eigenvectors, positive_singular_values, positive_eigenvectors

print(svd(np.array([[1, 2], [3, 4]])))