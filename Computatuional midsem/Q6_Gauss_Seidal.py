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

solution = ml.gauss_seidel(A, B, x0, eps)
print("\nSolutions are=",solution)


'''
The exact output:

The given A matrix is:  [[-2.0, 0.0, 0.0, -1.0, 0.0, 0.5], [0.0, 4.0, 0.5, 0.0, 1.0, 0.0], [0.0, 0.5, 1.5, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, -2.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0, -2.5, 0.0], [0.5, 0.0, 0.0, 1.0, 0.0, -3.75]]

The given B matrix is: [[-1.0], [0.0], [2.75], [2.5], [-3.0], [2.0]]

The guess matrix is:  [[1.0], [1.0], [1.0], [1.0], [1.0], [1.0]]
xk1 [[0], [0], [0], [0], [0], [0]]
c= 12

Solutions are= [[1.499998817301484], [-0.5000000000000033], [2.000000000000001], [-2.4999994084139723], [0.9999999999999987], [-0.9999999999368614]]
'''