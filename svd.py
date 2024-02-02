import numpy.matlib 
import numpy as np 
import math
A = np.array([[1, -1, 3], 
              [3, 1, 1]])

print(A[0])
print("Matrix A:\n", A)
AT = A.transpose()
print("Matrix AT:\n", AT)

ATA = np.dot(AT, A)
AAT = np.dot(A, AT)
print("Matrix AAT:\n", ATA)
eigenvalues, eigenvectors = np.linalg.eig(ATA)

#Calculate sigma
sigma = np.zeros_like(A, dtype=np.float64)
msigma = np.sqrt(eigenvalues)
index = 0

for i in range(len(A)):
    sigma[i][index] = msigma[index]
    index += 1
print("Sigma matrix:\n", sigma)

#Matrix VT
V = eigenvectors
VT = eigenvectors.transpose()
print("\nMatrix VT:\n", VT)

#Calculate Matrix U
#AV = sigmaU => U = AV/sigma => ui = A*vi/sigmai
U = np.empty((len(A), len(A)))
index = 0
for i in range(len(A)):
    U[:, i] = np.divide(np.dot(A, V[:,i]), msigma[i]) #vector column
print("matrix U:\n",U)
print(U.T)
#check the result of three matrices
print("Matrix complete after applying SVD:\n", np.dot(np.dot(U, sigma), VT))

