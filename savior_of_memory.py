import numpy as np
import sys
import scipy.linalg as la
from imageio.v2 import imread
from matplotlib import pyplot as plt
from PIL import Image

def svd(A: np.array, error_tolerance: float=10**-50):
    #Get the eigenvalues and eigenvectors of (A^H)A
    eigenvalues, eigenvectors = la.eigh(np.dot(A.conj().T, A))
    print(min(eigenvalues), max(eigenvalues))
    
    #Calculate the singular values
    singular_values = eigenvalues ** 0.5

    #Count the number of nonzero singular values
    rank = len([singular_value for singular_value in singular_values if singular_value > error_tolerance])

    #Keep only the positive singular values
    positive_singular_values = singular_values[:rank]

    #Keep only the corresponding eigenvectors
    positive_eigenvectors = eigenvectors[:, :rank]

    #Construct U with array broadcasting
    U = np.dot(A, np.dot(positive_eigenvectors, np.diag(1/positive_singular_values)))

    return U, positive_singular_values, positive_eigenvectors.conj().T

def truncated_svd(A: np.array, s: int):
    U, singular_values, VH = np.linalg.svd(A)
    print(singular_values)

    #U_truncated should be m x s, V_truncated should be n x s, and sigma_truncated should be s x s
    U_truncated = U[:, :s]
    singular_values_truncated = singular_values[:s]
    VH_truncated = VH[:s, :]
    return U_truncated, singular_values_truncated, VH_truncated

def compress_image(image_file: str):
    #Read the image as an array
    image = imread(image_file)
    if len(image.shape) > 2:
        print("Image not grayscale...exiting")
        exit(0)
    U, singular_values, VH = truncated_svd(image, int(input(f"This image has {image.shape[0]} singular values. Enter the number of singular values to retain: ")))
    compressed_matrix = np.dot(U, np.dot(np.diag(singular_values), VH))
    data = Image.fromarray(compressed_matrix, mode='L')
    output_file = f"{image_file.split('.')[0]}_compressed.jpg"
    data.save(output_file)
    print(f"Compressed image saved as {output_file}")


'''
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
'''

compress_image(sys.argv[1])