import numpy as np
from scipy.linalg import dft

np.set_printoptions(linewidth=160)

n = 12
m = 6

l = np.zeros([n,n])
for k in range(m):
    l[k,k] = 1

# a = np.fft.dft(l)

F = dft(n)

A = np.conj(F) @ l @ F 

def print_nice(A):
    print(np.real(np.around(A, 1)))
    print()
    # print(np.imag(np.around(A, 1)))

print_nice(A)

a = np.real(np.around(A, 1))[0]

B = np.zeros([n,n])
for k in range(n):
    for j in range(n):
        B[k,j] = a[(n + j - k) % n]

print_nice(B)