import my_library_mid as ml


A = ml.matrix_read("Q4_msfit.txt")

print("A",A)

B = ml.transpose(A)
print("A",B)

time = [B[0]]
print("A",time)
time = ml.transpose(time)
count = [B[1]]
count = ml.transpose(count)

err = [B[2]]
err = ml.transpose(err)