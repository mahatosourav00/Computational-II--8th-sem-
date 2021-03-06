import math
import random
import time
import matplotlib.pyplot as plt




import math

def linear_regression(X,Y, sig):
    
    a = 0
    b = 0
    chi2 = 0
    S = 0
    Sx = 0
    Sy = 0
    Sxx = 0
    Syy = 0
    Sxy = 0
    err_a = 0
    err_b = 0
    sig2_a = 0
    sig2_b = 0
    Y1 = [0 for i in range(len(X))]
    
    for i in range(len(X)):
        
        S = S + 1/(sig[i]**2)
        Sx = Sx + X[i]/(sig[i]**2)
        Sy = Sy + Y[i]/(sig[i]**2)
        Sxx = Sxx + (X[i]**2)/(sig[i]**2)
        Sxy = Sxy + (X[i]*Y[i])/(sig[i]**2)
        Syy = Syy + (Y[i]**2)/(sig[i]**2)
        
    delta = S*Sxx - (Sx**2)
    a = (Sxx*Sy - Sx*Sxy)/delta
    b = (S*Sxy - Sx*Sy)/delta
    
    for i in range(len(X)):
        Y1[i] = a + b * X[i]
        chi2 = chi2 + ((Y[i] - Y1[i])/sig[i])**2
    
    quality = chi2/(len(X)-2)
    covab = -Sx/delta
    sig2_a = Sxx/delta
    err_a = math.sqrt(sig2_a)
    sig2_b = S/delta
    err_b = math.sqrt(sig2_b)
    r2 = Sxy**2/(Sxx*Syy)
    return a,b, covab, err_a, err_b






def jackknife(A):
    n = len(A[0])
    yj_mean = make_matrix(len(A),n) #means of data set
    yj_mean2 = make_matrix(len(A),n) #squares of means
    #print("yj_mean=",yj_mean)
    #print("yj_mean2=",yj_mean2)
    

    for i in range(len(A)):
        yj = []
        yj2 = []
        
        for k in range(len(A[0])):
            sum_y = 0.0
            for j in range(len(A[0])):
                if j != k:
                    sum_y = sum_y + A[i][j]      

            yj.append(sum_y/(n-1))
            yj2.append((sum_y/(n-1))*(sum_y/(n-1)))
       
        yj_mean[i]=yj
        yj_mean2[i]=yj2


    #print("yj_mean=",yj_mean)
    #print("yj_mean2=",yj_mean2)
        
    yjk = make_matrix(len(A),1) #mean of yj_mean
    yj2_mean = make_matrix(len(A),1) #mean of yj_mean2
    for i in range(len(A)):
        
        sum_y = 0.0
        sum_y2 = 0.0
        for j in range(n):
            sum_y = sum_y + yj_mean[i][j]
            sum_y2 = sum_y2 + yj_mean2[i][j]
        yjk[i][0] = (sum_y/n)
        yj2_mean[i][0] = (sum_y2/n)
        
    yjk2 = make_matrix(len(A),1)
    #print("yjk=",yjk)
    for i in range(len(yjk)):
        yjk2[i][0] = (yjk[i][0]*yjk[i][0])
    #print("yjk2=",yjk2)
   # print("yj2mean=",yj2_mean)
    sig_jk2 = matrix_substraction(yj2_mean,yjk2)
    
    sig = scaler_matrix_multiplication((n-1),sig_jk2)
    print("yjk",yjk)
    #print("sigjk2",sig_jk2)
    print("sig",sig)
    return yjk, sig





def diagonal_elems(A):
    diag = make_matrix(1,len(A))
    for i in range(len(A)):
        diag[0][i] = A[i][i]
    return diag

def jacobi_eigen(A, eps):
    
    
    def max_offdiag(A):
        maxd = 0.0
        for i in range(len(A)-1):
            for j in range(i+1,len(A)):
                if abs(A[i][j])>=maxd:
                    maxd = abs(A[i][j])
                    k=i
                    l=j
        return maxd,k,l
    
    def rotate(A,P,k,l):

        diff = A[l][l] - A[k][k]
        if abs(A[k][l]) < abs(diff)*1.0e-36:
            t = A[k][l]/diff
        else:
            phi = diff/(2.0*A[k][l])
            t = 1.0/(abs(phi) + math.sqrt(phi**2 + 1.0))
            if phi < 0.0:
                t = -t
        c = 1.0/math.sqrt(t**2 + 1.0)
        s = t*c
        #print("s",s)
        #print("c",c)
        tau = s/(1.0 + c)
    
        store = A[k][l]
        A[k][l] = 0.0
        A[k][k] = A[k][k] - t * store
        A[l][l] = A[l][l] + t * store
    
        for i in range(k):
            store = A[i][k]
            A[i][k] = store - s * (A[i][l] + tau * store)
            A[i][l] = A[i][i] + s * (store - tau * A[i][l])
    
        for i in range(k+1,l):
            store = A[k][i]
            A[k][i] = store - s * (A[i][l] + tau * A[k][i])
            A[i][l] = A[i][l] - s * (store - tau * A[i][l])
            
        for i in range(l+1,len(A)):
            store = A[k][i]
            A[k][i] = store - s * (A[l][i] + tau * store)
            A[l][i] = A[l][i] + s * (store - tau * A[l][i])
            
        for i in range(len(A)):
            store = P[i][k]
            P[i][k] = store - s * (P[i][l] + tau * P[i][k])
            P[i][l] = P[i][l] + s * (store - tau * P[i][l])
            
    max_rotation = 5*(len(A)**2)
    P = unit_matrix(len(A))
    for i in range(max_rotation):
        maxd, k, l = max_offdiag(A)
        if maxd < eps:
            diag = diagonal_elems(A)
            
            print("Eigenvalues = ", diagonal_elems(A))
            print("Eigenvectors =", P)
            return diag, P
        rotate(A,P,k,l)
    print("There is no convergence!")
    








def frob_norm(A):
    sum = 0
    for i in range(len(A)):
        for j in range(len(A[i])):
            sum = sum + (A[i][j]**2)
    return math.sqrt(sum)

def power_normalize(A):
    max = 0
    for i in range(len(A)):
        if max <= A[i][0]:
            max = A[i][0]
    normA = scaler_matrix_division(max,A)
    return normA


def power_method(A, x0, eps):
    i = 0
    lam0 = 1
    lam1 = 0
    while abs(lam1-lam0) >= eps:
        print("error=",abs(lam1-lam0))
        if i != 0:
            lam0 = lam1
        
        Ax0 = matrix_multiplication(A,x0)
        AAx0 = matrix_multiplication(A,Ax0)
        #print("Ax0=",Ax0)
        #print("AAx0=",AAx0)
        dotU = inner_product(AAx0,Ax0)
        dotL = inner_product(Ax0,Ax0)
        #print("U=",dotU)
        #print("L=",dotL)
        lam1 = dotU/dotL
        
        x0 = Ax0
        i = i+1
        print("i=",i)
        
        print("eigenvalue=",lam1)
        ev = power_normalize(x0)
        print ("eigenvector=",ev)
    return lam1, ev





def conju_norm(A):
    sum=0
    for i in range(len(A)):
        sum = sum + abs(A[i][0])
    return sum

def inner_product(A,B):

    AT = transpose(A)

    C = matrix_multiplication(AT, B)

    return C[0][0]


def conjugate_gradient(A, B, x0, eps):
    #r0 = make_matrix(len(A), 1)
    xk = x0
    
    #r0=b-Ax0
    Ax0 = matrix_multiplication(A, x0)
    rk = matrix_substraction(B, Ax0)
    #print("rk",rk)
    i = 0
    dk = rk
    #print("dk",dk)
    
    
    while conju_norm(rk)>=eps and i in range(len(A)):
        adk = matrix_multiplication(A,dk)
        #print("adk=",adk)
        rkrk = inner_product(rk, rk)
        #print("rkrk = ", rkrk)
        alpha = rkrk/inner_product(dk, adk)
        #print("alpha = ",alpha)
        xk = matrix_addition(xk, scaler_matrix_multiplication(alpha, dk))
        #print("xk1=",xk)
        rk = matrix_substraction(rk, scaler_matrix_multiplication(alpha, adk))
        #print("rk1=",rk)
        beta = inner_product(rk, rk)/rkrk
        dk = matrix_addition(rk, scaler_matrix_multiplication(beta, dk))
        
        i = i+1
        print("norm=",conju_norm(rk))
        print("i=",i)
    return xk



'''
def conju_norm(A):
    sum=0
    for i in range(len(A)):
        sum = sum + abs(A[i])
    return sum        
        

def conjugate_gradient(A, B, x0, eps):
    xk = x0
    rk = B- np.dot(A, x0)
    dk = rk
    i = 0
    print("norm=",conju_norm(rk))
    while i in range(len(A)):
        adk = np.dot(A, dk)
        rkrk = np.dot(rk, rk)
        
        alpha = rkrk / np.dot(dk, adk)
        print("xk0=", xk)
        xk = xk + alpha * dk
        print("xk1=", xk)
        rk = rk - alpha * adk
        if conju_norm(rk)<=eps:
            break
        else:
            beta = np.dot(rk, rk) / rkrk
            print("dk0=",dk)
            dk = rk + beta * dk
            print("dk1=",dk)
            i= i + 1
            print("i=",i)
            print("norm=",conju_norm(rk))
    return xk
    

'''

def gauss_seidel(A, B, eps):
    
    # Check: A should have zero on diagonals
    for i in range(len(A)):
        if A[i][i] == 0:
            return ("Main diagnal should not have zero!")

        
    xk0 = make_matrix(len(A),1)
    xk1 = make_matrix(len(A),1)
    print("Guess the x matrix of length",len(A))
    for i in range(len(xk0)):
        for j in range(len(xk0[i])):
            xk0[i][j]=float(input("element:"))
            
    print("xk1",xk1)
    c=0
    while inf_norm(xk1,xk0) >= eps:
        
        if c!=0:
                for i in range(len(xk1)):
                    for j in range(len(xk1[i])):
                        xk0[i][j]=xk1[i][j]
        for i in range(len(A)):
            sum1 = 0
            sum2 = 0
            for j in range(i+1,len(A[i])):
                sum2 = sum2 + (A[i][j]*xk0[j][0])
            for j in range(0,i):
                sum1 = sum1 + (A[i][j]*xk1[j][0])
            xk1[i][0] = (1/A[i][i])*(B[i][0]-sum1-sum2)
            
        c=c+1
    print("c=",c)
        
    return xk1







def inf_norm(X,Y):
    max=0

    sum=0
    for i in range(len(X)):
        for j in range(len(X[i])):
            diff = abs(X[i][j]-Y[i][j])
            
            sum = sum + diff
            
        if sum>max:
            max = sum
    return max
        

def jacobi(A, B, eps):
    
    # Check: A should have zero on diagonals
    sumdiag = 0
    sumother = 0
    for i in range(len(A)):
        sumdiag = sumdiag + A[i][i]
        for j in range(len(A[i])):
            if i != j:
                sumother = sumother + A[i][j]
        if A[i][i] == 0:
            return ("Main diagnal should not have zero!")
    print("sumdiag",sumdiag)
    print("sumother",sumother)
    if sumdiag<=sumother:
        return ("Sum of diagonal must be dominant!")
        
    xk0 = make_matrix(len(A),1)
    xk1 = make_matrix(len(A),1)
    print("Guess the x matrix of length",len(A))
    for i in range(len(xk0)):
        for j in range(len(xk0[i])):
            xk0[i][j]=float(input("element:"))
            

    c=0
    while inf_norm(xk1,xk0) >= eps:
        
        if c!=0:
                for i in range(len(xk1)):
                    for j in range(len(xk1[i])):
                        xk0[i][j]=xk1[i][j]
        for i in range(len(A)):
            sum = 0
            for j in range(len(A[i])):
                if j!=i:
                    sum = sum + (A[i][j]*xk0[j][0])
            xk1[i][0] = (1/A[i][i])*(B[i][0]-sum)
        c=c+1
    print("c=",c)
        
    return xk1
            
                    
    
    








def partial_pivot_solution(A):
    # row loop for checking 0 on diagonal positions
    for r1 in range(len(A)-1):
        if abs(A[r1][r1]) == 0:
            # row loop for finding suitable row for interchanging
            for r2 in range(r1 + 1, len(A)):
                # row interchange
                if A[r2][r1] > A[r1][r1]:
                    a1 = A[r1]
                    A[r1] = A[r2]
                    A[r2] = a1
    return A



def partial_pivot_inverse(A, B):

    # row loop for checking 0 on diagonal positions
    for r1 in range(len(A)-1):
        if abs(A[r1][r1]) == 0:
            # row loop for finding suitable row for interchanging
            for r2 in range(r1 + 1, len(A)):
                # row interchange
                if A[r2][r1] > A[r1][r1]:
                    a1 = A[r1]
                    A[r1] = A[r2]
                    A[r2] = a1
                    b1 = B[r1]
                    B[r1] = B[r2]
                    B[r2] = b1
                    
    return A, B


def gauss_jordan_solution(A):
    #row loop
    for r1 in range(len(A)):
        #performing pivoting
        partial_pivot_solution(A)
        pivot = A[r1][r1]
        #column loop
        for c1 in range(len(A[r1])):
            A[r1][c1] = A[r1][c1]/pivot
        for r2 in range(len(A)):
            if r2 == r1 or A[r2][r1] == 0:
                pass
            else:
                factor = A[r2][r1]
                for c1 in range(len(A[r2])):
                    A[r2][c1] = A[r2][c1] - factor * A[r1][c1]    
    return A


def gauss_jordan_inverse(A):
    #row loop
    if len(A) != len(A[1]):
        print("Matrix need to be square matrix")
    else:
        B = unit_matrix(len(A))
        for r1 in range(len(A)):
            # performing pivoting
            partial_pivot_inverse(A, B)
            pivot = A[r1][r1]
            #column loop
            for c1 in range(len(A[r1])):
                A[r1][c1] = A[r1][c1]/pivot
            for c2 in range(len(B[r1])):
                    B[r1][c2] = B[r1][c2] / pivot
            for r2 in range(len(A)):
                if r2 == r1 or A[r2][r1] == 0:
                    pass
                else:
                    factor = A[r2][r1]
                    for c1 in range(len(A[r2])):
                        A[r2][c1] = A[r2][c1] - factor * A[r1][c1]
                    for c2 in range(len(B[r1])):
                        B[r2][c2] = B[r2][c2] - factor * B[r1][c2]
        return B



def lu_decomposition(A):
    n = len(A)

    #perform LU Decomposition
    #Both Upper and Lower triangular matrix will be stored on A matrix together
    for j in range(n):

        # upper trianguar matrix
        for i in range(j+1):
            sum = 0
            for k in range(i):
                sum = sum + A[i][k] * A[k][j]
            #store to A matrix
            A[i][j] = A[i][j] - sum

        #lower triangular matrix
        for i in range(j+1, n):
            sum = 0
            for k in range(j):
                sum = sum + A[i][k] * A[k][j]
            # store to M matrix
            A[i][j] = (A[i][j] - sum)/A[j][j]

    return (A)



def forward_substitution(L, B):

    #creat Y matrix
    Y = [[0] for i in range(len(L))]

    # calculate Y matrix
    for i in range(len(L)):
        sum = 0
        for j in range (i+1):
            if i == j:
                pass
            else:
                sum = sum + L[i][j] * Y[j][0]
        Y[i][0] = (B[i][0] - sum)

    #return the calculated Y matrix
    return (Y)



def backward_substituition(U, Y):
    # creat x matrix
    X = [[0] for i in range(len(U))]

    # calculate x matrix
    for i in range(len(U) - 1, -1, -1):
        sum = 0
        for j in range(len(U) - 1, i - 1, -1):
            sum = sum + U[i][j] * X[j][0]
        X[i][0] = (Y[i][0] - sum) / U[i][i]

    # return the calculated x matrix
    return (X)




def transpose(A):
    #if a 1D array, convert to a 2D array = matrix
    #if not isinstance(A[0],list):
       # A = [A]
 
    #Get dimensions
    r = len(A)
    c = len(A[0])

    #AT is zeros matrix with transposed dimensions
    AT = make_matrix(c, r)

    #Copy values from A to it's transpose AT
    for i in range(r):
        for j in range(c):
            AT[j][i] = A[i][j]

    return AT


def scaler_matrix_multiplication(c,A):
    cA = make_matrix(len(A), len(A[0]))
    for i in range(len(A)):
        for j in range(len(A[i])):
            cA[i][j] = c * A[i][j]
    return cA
    

def scaler_matrix_division(c,A):
    cA = make_matrix(len(A), len(A[0]))
    for i in range(len(A)):
        for j in range(len(A[i])):
            cA[i][j] = A[i][j]/c
    return cA


def matrix_multiplication(A, B):
    AB =  [[0.0 for j in range(len(B[0]))] for i in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[i])):
            add = 0
            for k in range(len(A[i])):
                multiply = (A[i][k] * B[k][j])
                add = add + multiply
            AB[i][j] = add
    return (AB)





def matrix_addition(A, B):
    
    ra = len(A)
    ca = len(A[0])
    rb = len(B)
    cb = len(B[0])
    
    if ra != rb or ca != cb:
        raise ArithmeticError('Matrices are NOT of the same dimensions!.')
    
    C = make_matrix(ra, cb)
    
    for i in range(ra):
        for j in range(cb):
            C[i][j]=A[i][j] + B[i][j]
    return C

def matrix_substraction(A, B):
    
    ra = len(A)
    ca = len(A[0])
    rb = len(B)
    cb = len(B[0])
    
    if ra != rb or ca != cb:
        raise ArithmeticError('Matrices are NOT of the same dimensions!.')
    
    C = make_matrix(ra, cb)
    
    for i in range(ra):
        for j in range(cb):
            C[i][j]=A[i][j] - B[i][j]
    return C



def matrix_read(B):
    #read the matrix text files
    a = open(B)
    A = []
    #A matrix
    for i in a:
        A.append([int(j) for j in i.split()])
    return (A)


def matrix_copy(A):
    B = make_matrix(len(A), len(A[0]))
    for i in range(len(A)):
        for j in range(len(A[i])):
            B[i][j] = A[i][j]
    return B



def matrix_print(A):
    for i in A:
        for j in i:
            print(j, end='  ')
        print()

def unit_matrix(A):
    B = [[0 for x in range(A)] for y in range(A)]
    for i in range(len(B)):
        for j in range(len(B[i])):
            if i==j:
                B[i][j]=1
    return B


def make_matrix(N, M):
    I = [[0 for x in range(M)] for y in range(N)]
    return I