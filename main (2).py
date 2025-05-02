import numpy as np

def gaussian_elimination_with_pivoting(A, b):
    """
    Solves the linear system Ax = b using Gaussian elimination with partial pivoting.

    Args:
        A (numpy.ndarray): The coefficient matrix.
        b (numpy.ndarray): The constant vector.

    Returns:
        numpy.ndarray: The solution vector x.
    """
    n = A.shape[0]
    Ab = np.concatenate((A, b.reshape(-1, 1)), axis=1)  # Augmented matrix

    for i in range(n):
        # Partial pivoting: find the row with the largest absolute value in the current column and below
        pivot_row = i
        for j in range(i + 1, n):
            if abs(Ab[j, i]) > abs(Ab[pivot_row, i]):
                pivot_row = j

        # Swap the current row with the pivot row
        Ab[[i, pivot_row]] = Ab[[pivot_row, i]]

        # Elimination
        pivot = Ab[i, i]
        if pivot == 0:
            raise ValueError("Matrix is singular, no unique solution or multiple solutions exist.")

        for j in range(i + 1, n):
            factor = Ab[j, i] / pivot
            Ab[j, i:] = Ab[j, i:] - factor * Ab[i, i:]

    # Back-substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i + 1:n], x[i + 1:n])) / Ab[i, i]

    return x

# 1. Solve the following system using Gaussian elimination and pivoting
A1 = np.array([
    [1.19, 2.11, -100, 1],
    [14.2, -0.112, 12.2, -1],
    [0, 100, -99.9, 1],
    [15.3, 0.110, -13.1, -1]
], dtype=float)
b1 = np.array([1.12, 3.44, 2.15, 4.16], dtype=float)

try:
    x1 = gaussian_elimination_with_pivoting(A1, b1)
    print("1. Solution using Gaussian elimination with pivoting:")
    print(f"x1 = {x1[0]:.4f}, x2 = {x1[1]:.4f}, x3 = {x1[2]:.4f}, x4 = {x1[3]:.4f}")
except ValueError as e:
    print(f"1. Error solving the system: {e}")

def inverse_of_matrix(A):
    """
    Calculates the inverse of matrix A.

    Args:
        A (numpy.ndarray): The invertible square matrix.

    Returns:
        numpy.ndarray: The inverse of matrix A.
    """
    try:
        A_inv = np.linalg.inv(A)
        return A_inv
    except np.linalg.LinAlgError:
        print("Matrix is singular, no inverse exists.")
        return None

# 2. Find the inverse of matrix A
A2 = np.array([
    [4, 1, -1, 0],
    [1, 3, -1, 0],
    [-1, -1, 6, 2],
    [0, 0, 2, 5]
], dtype=float)

A2_inv = inverse_of_matrix(A2)
if A2_inv is not None:
    print("\n2. Inverse of matrix A:")
    print(A2_inv)

def crout_factorization_tridiagonal(A, b):
    """
    Solves a tridiagonal linear system Ax = b using Crout factorization.

    Args:
        A (numpy.ndarray): The tridiagonal coefficient matrix.
        b (numpy.ndarray): The constant vector.

    Returns:
        numpy.ndarray: The solution vector x.
    """
    n = A.shape[0]
    if A.shape[1] != n or n != b.shape[0]:
        raise ValueError("Matrix A must be square, and its dimensions must match the length of vector b.")

    # Extract the diagonals
    a = np.diag(A, k=-1)  # Lower diagonal
    b_diag = np.diag(A)    # Main diagonal
    c = np.diag(A, k=1)   # Upper diagonal

    # Initialize the diagonal elements of L and U
    alpha = np.zeros(n)
    beta = np.zeros(n)
    gamma = np.zeros(n)

    # Crout factorization
    alpha[0] = b_diag[0]
    beta[0] = c[0] / alpha[0]

    for i in range(1, n):
        alpha[i] = b_diag[i] - a[i-1] * beta[i-1]
        if i < n - 1:
            beta[i] = c[i] / alpha[i]

    # Forward substitution Ly = b
    y = np.zeros(n)
    y[0] = b[0] / alpha[0]
    for i in range(1, n):
        y[i] = (b[i] - a[i-1] * y[i-1]) / alpha[i]

    # Backward substitution Ux = y
    x = np.zeros(n)
    x[n-1] = y[n-1]
    for i in range(n - 2, -1, -1):
        x[i] = y[i] - beta[i] * x[i+1]

    return x

# 3. Solve the tridiagonal system using Crout factorization
A3 = np.array([
    [3, -1, 0, 0],
    [-1, 3, -1, 0],
    [0, -1, 3, -1],
    [0, 0, -1, 3]
], dtype=float)
b3 = np.array([2, 3, 4, 1], dtype=float)

try:
    x3 = crout_factorization_tridiagonal(A3, b3)
    print("\n3. Solution of the tridiagonal system using Crout factorization:")
    print(f"x1 = {x3[0]:.4f}, x2 = {x3[1]:.4f}, x3 = {x3[2]:.4f}, x4 = {x3[3]:.4f}")
except ValueError as e:
    print(f"3. Error solving the system: {e}")