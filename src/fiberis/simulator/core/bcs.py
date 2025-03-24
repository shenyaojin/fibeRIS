# This script defined the boundary conditions for PDS simulator1d
# Shenyao Jin, 02/18/2025
from io import RawIOBase

import numpy as np
from keyring import set_keyring


class BoundaryCondition:

    def __init__(self, type=None, value=None, idx=None):
        self.type = type
        self.value = value
        self.matA = None # The matrix A at left hand side
        self.vecB = None  # The vector B at right hand side
        self.idx = idx

    def set_matrix(self, matA, vecB):
        self.matA = matA
        self.vecB = vecB

        # Self check
        shapeA = np.shape(matA)
        shapeB = np.shape(vecB)

        if shapeA[0] != shapeA[1]:
            raise ValueError('Matrix A is not square')

        # Check vector B is a vector
        if len(shapeB) != 1:
            raise ValueError('Vector B is not a vector')

        # Check the size of mat A and vec B
        if shapeA[0] != shapeB[0]:
            raise ValueError('The size of matrix A and vector B does not match')

    # Apply the boudary condition to the matrix and vector -> Ax = B
    # At the idx position
    def apply(self):

        if self.type == 'Dirichlet':
            self.matA[self.idx, :] = 0
            self.matA[self.idx, self.idx] = 1

            self.vecB[self.idx] = self.value
        elif self.type == 'Neumann':
            self.matA[self.idx, :] = 0
            self.matA[self.idx, self.idx] = 1
            if self.idx == 0:
                # Change the idx+1;
                self.matA[self.idx, self.idx+1] = -1
            else:
                self.matA[self.idx, self.idx-1] = -1

            self.vecB[self.idx] = self.value

        elif self.type == 'PML':
            # implement the PML boundary condition
            pass
        # Add your own BCs here
        else:
            raise ValueError('Unknown BC type')