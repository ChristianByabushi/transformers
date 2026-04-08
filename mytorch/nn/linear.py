import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        """
        Initialize the weights and biases with zeros
        W shape: (out_features, in_features)
        b shape: (out_features,)  # Changed from (out_features, 1) to match PyTorch
        """
        # DO NOT MODIFY
        self.W = np.zeros((out_features, in_features))
        self.b = np.zeros(out_features)


    def init_weights(self, W, b):
        """
        Initialize the weights and biases with the given values.
        """
        # DO NOT MODIFY
        self.W = W
        self.b = b


# it seems to me that we are receiving a lot of values in 3d, 4d..
# but our aim is to to apply all them with same weights,
# so we reshape the batch and the sequences in a such way that we 
# apply on them the same weigths.

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (*, in_features)
        :return: Output Z with shape (*, out_features)
        
        Handles arbitrary batch dimensions like PyTorch 
        
        """
        # TODO: Implement forward pass
        
        # Store input for backward pass
        self.A = A

        self.input_shape = A.shape

        # e.g. (B, T, in_features) -> (B*T, in_features)
        A_2d = A.reshape(-1, A.shape[-1])

        self.A = A_2d # store flattened version for backward 

        
        
        # Standard affine: Z = A @ W^T + b
        Z_2_d = A_2d @ self.W.T + self.b 

        # restore original leading dims, replace last dim with out_features
        # e.g *(3,2) >> (3,2)
        Z = Z_2d.reshape(*self.input_shape[:-1], self.W.shape[0]) 

        return Z

    def backward(self, dLdZ):
        """

        :param dLdZ: Gradient of loss wrt output Z (*, out_features)
        :return: Gradient of loss wrt input A (*, in_features)
        
        """

        
        dLdZ_2d = dLdz.reshape(-1, dLdZ.shape[-1])  

        A_2d = self.A 

        # self.dLdW = dLdZ.T @ self.A
        self.dLdW = dLdZ_2d.T @ A_2d 

        self.dLdb = np.sum(dLdZ_2d, axis=0) 

        self.dLA_2d = dLdZ_2d @ self.W
        # self.dLdA = dLdZ @ self.W
        
        self.dLdA = dLdZ_2d.reshape(self.input_shape) 

        return self.dLdA
