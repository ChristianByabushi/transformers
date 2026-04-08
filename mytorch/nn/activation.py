import numpy as np


class Softmax:
    """
    A generic Softmax activation function that can be used for any dimension.
    """
    def __init__(self, dim=-1):
        """
        :param dim: Dimension along which to compute softmax (default: -1, last dimension)
        DO NOT MODIFY
        """
        self.dim = dim

    def forward(self, Z):
        """
        :param Z: Data Z (*) to apply activation function to input Z.
        :return: Output returns the computed output A (*).
        """
        if self.dim > len(Z.shape) or self.dim < -len(Z.shape):
            raise ValueError("Dimension to apply softmax to is greater than the number of dimensions in Z")
        
        # TODO: Implement forward pass
        # moveaxis (array, source, destination) -> the place 
        Z_moved = np.moveaxis(Z, self.dim,-1)
        original_shape = Z_moved.shape 


        # Flatten to 2D
        Z_2d = Z_moved.reshape(-1, Z_moved.shape[-1]) 

        # Numerically stable softmax, # keepdims [2] not 2
        Z_2d -= np.max(Z_2d, axis=1, keepdims=True) 

        exp_Z = np.exp(Z_2d) 

        A_2d = exp_Z / np.sum(exp_Z, axis=1, keepdims=True) 

        self.A = np.moveaxis(A_2d.reshape(original_shape),-1,self.dim)

        return self.A

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt output
        :return: Gradient of loss with respect to activation input
        """
        
        dLdA_moved = np.moveaxis(dLdA, self.dim, -1) 
        A_moved = np.moveaxis(self.A, self.dim, -1) 
        original_shape = A_moved.shape

        dLdA_2d = dLdA_moved.reshape(-1, dLdA_moved.shape[-1]) 
        A2_d = A_moved.reshape(-1, A_moved.shape[-1]) 

        dLdZ = A_2d * (dLdA_2d - np.sum(dLdA_2d * A_2d, axis=1, keepdims=True))

        dLdZ = np.moveaxis(dLdZ_2d.reshape(orignial_shape),-1, self.dim)

        return dLdZ






 

    