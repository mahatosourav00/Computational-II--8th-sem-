
import my_library_mid as ml


A = ml.matrix_read("Q6_matrixA_msmatinv.txt")
B = ml.matrix_read("Q6_matrixB_msmatinv.txt")


#guess

x0 = ml.make_matrix(len(A),1)

for i in range(len(A)):
    x0[i][0] = 1.0



print("The given A matrix is: ", A)

print("\nThe given B matrix is:", B)

print("\nThe guess matrix is: ", x0)

eps = 0.00001
solution = ml.jacobi(A,B,x0,eps)
print("\nSolutions are=",solution)


'''
The exact output:

The given A matrix is:  [[-2.0, 0.0, 0.0, -1.0, 0.0, 0.5], [0.0, 4.0, 0.5, 0.0, 1.0, 0.0], [0.0, 0.5, 1.5, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, -2.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0, -2.5, 0.0], [0.5, 0.0, 0.0, 1.0, 0.0, -3.75]]

The given B matrix is: [[-1.0], [0.0], [2.75], [2.5], [-3.0], [2.0]]

The guess matrix is:  [[1.0], [1.0], [1.0], [1.0], [1.0], [1.0]]

Solutions are= [[1.500001092845368], [-0.5], [2.0], [-2.4999987702125672], [1.0], [-1.000000663125626]]


'''