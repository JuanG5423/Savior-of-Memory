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

    #Construct U with array broadcasting
    U = np.dot(A, np.dot(positive_eigenvectors, np.diag(1/positive_singular_values)))

    return U, positive_singular_values, positive_eigenvectors.conj().T

def truncated_svd(A: np.array, s: int):
    U, singular_values, VH = svd(A)

    #U_truncated should be m x s, V_truncated should be n x s, and sigma_truncated should be s x s
    U_truncated = U[:, :s]
    singular_values_truncated = singular_values[:s]
    V_truncated = VH[:, :s]
    return U_truncated, singular_values_truncated, V_truncated.conj().T

A = np.array([[1, 2, 3], [3, 4, 5]])
U, singular_values, VH = svd(A)
print(f"U: \n{U}")
print(f"Singular Values: \n{singular_values}")
print(f"VH: \n{VH}")
print(f"Original Matrix: \n{np.dot(U, np.dot(np.diag(singular_values), VH))}")

U_truncated, singular_values_truncated, VH_truncated = truncated_svd(A, 2)
print(f"U_truncated: \n{U_truncated}")
print(f"Singular Values Truncated: \n{singular_values_truncated}")
print(f"VH_truncated: \n{VH_truncated}")
print(f"Reconstructed Matrix: \n{np.dot(U_truncated, np.dot(np.diag(singular_values_truncated), VH_truncated))}")
