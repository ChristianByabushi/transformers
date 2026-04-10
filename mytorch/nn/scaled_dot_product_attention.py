import numpy as np
from .activation import Softmax

class ScaledDotProductAttention:
    """
    Scaled Dot Product Attention
    """ 
    def __init__(self):
        '''
        Initialize the ScaledDotProductAttention class.
        '''
        # Initialize your softmax layer
        # What dimension should you pass to the softmax constructor?
        self.eps = 1e10 # DO NOT MODIFY
        self.softmax = Softmax(dim=-1)
        
    
    def forward(self, Q, K, V, mask=None):
        """
        :param Q: Query matrix of shape (N, ..., H, L, E) where L is target sequence length
        :param K: Key matrix of shape (N, ..., H, S, E) where S is source sequence length
        :param V: Value matrix of shape (N, ..., H, S, Ev) where Ev is value dimension
        :param mask: Boolean mask matrix of shape (N, ..., H, L, S) or broadcastable shape where 1/True indicates a position to ignore
        :return: Output matrix of shape (N, ..., H, L, Ev)
        """

        self.Q = Q
        self.K = K
        self.V = V

        # TODO: Implement forward pass
        dk = Q.shape[-1]
        # Calculate attention scores we transpose K (...., S,E) to K(....,E,S)
        scaled_dot_product = (Q @ K.swapaxes(-2,-1)) / np.sqrt(dk)
        
        # here the substraction is applied where Q1 cannot attend when mask is true in np.where(condition, return_value_if_true, return_value_if_false)
        # Apply mask before softmax if provided
        if mask is not None:
            scaled_dot_product = np.where(mask, scaled_dot_product - self.eps, scaled_dot_product)


        # Compute attention scores: 
        # # Think about which dimension you should apply Softmax
        self.attention_scores = self.softmax.forward(scaled_dot_product)

        # Calculate final output
        output = self.attention_scores @ V

        # Return final output
        return output
    
    def backward(self, d_output):
        """
        :param d_output: Gradient of loss wrt output of shape (N, ..., H, L, Ev)
        :return: Gradient of loss wrt input Q, K, V
        """
        # need to store Q,K,V in forward as well
        dk = self.Q.shape[-1]

        # dL/dV = A^T @ d_output --> (N,....,H,S,Ev)
        d_V = self.attention_scores.swapaxes(-2,-1) @ d_output
        

        # dL/dA = d_output @ V^T ->  (N,...,H,L,S)
        d_attention_scores = d_output @ self.V.swapaxes(-2,-1)


        # dL/dS = softmax backward 
        d_scaled_dot_product = self.softmax.backward(d_attention_scores)

        d_scaled_dot_product = d_scaled_dot_product/np.sqrt(dk)
        # dL/dQ = dL/dS @ K 
        d_Q = d_scaled_dot_product @ self.K 

        # dL/dK = dL/dS ^ T  @ Q
        d_K = d_scaled_dot_product.swapaxes(-2,-1) @ self.Q
        
        return d_Q, d_K, d_V

