import my_library_mid as ml


A = ml.matrix_read("Q5_mstrimat.txt")

#print("A", A)

x0 = ml.make_matrix(len(A),1)

for i in range(len(A)):
    x0[i][0] = 1.0
    
print("The given A matrix is: ", A)

#print("\nThe given B matrix is:", B)

print("\nThe guess matrix is: ", x0)

eps = 1e-5

e1, ev1 = ml.power_method(A,x0,eps)

print("\nThe largest eigenvalue is:",e1, "\nand corresponding eigenvector is:",ev1)
#for i in range(len(ev1)):
    #ev1[i][0] = ev1[i][0]/abs(ev1[i][0])

#print("ev1",ev1)

#ev1t = ml.transpose(ev1)

#B = A - lambda * I, here I = UT * U

I = [[0 for x in range(len(A))] for y in range(len(A))]
for i in range(len(A)):
    I[i][i] = 1
    
e1I = ml.scaler_matrix_multiplication(e1,I) 

for i in range(len(A)):
    x0[i][0] = 2.0

#print("I",e1I)
#print("A",A)
B = ml.matrix_substraction(A, e1I)


#print("B = A - lam*I matrix is",B)


e2, ev2not = ml.power_method(B,x0,eps)


div_ev = e1/e2

div_ev_ev1 = ml.scaler_matrix_multiplication(div_ev, ev1)

ev2 = ml.matrix_addition(div_ev_ev1, ev2not)

print("\nThe Second largest eigenvalue is:",e2, "\nand corresponding eigenvector is:",ev2)



'''
The given A matrix is:  [[2.0, -1.0, 0.0, 0.0, 0.0], [-1.0, 2.0, -1.0, 0.0, 0.0], [0.0, -1.0, 2.0, -1.0, 0.0], [0.0, 0.0, -1.0, 2.0, -1.0], [0.0, 0.0, 0.0, -1.0, 2.0]]

The guess matrix is:  [[1.0], [1.0], [1.0], [1.0], [1.0]]

The largest eigenvalue is: 3.7320486268542985 
and corresponding eigenvector is: [[0.5016849934385631], [-0.8669982351997433], [1.0], [-0.8669982351997433], [0.5016849934385631]]

The Second largest eigenvalue is: -3.464097536927689 
and corresponding eigenvector is: [[-0.03891913132201774], [1.8009941333673831], [-0.07735090801290068], [1.8009941333673831], [-0.03891913132201774]]

'''

