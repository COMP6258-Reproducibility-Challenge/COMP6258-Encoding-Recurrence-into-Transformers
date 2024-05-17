import numpy as np
import torch
from scipy.linalg import toeplitz
import torch.nn as nn

class REM(nn.Module):
    # initialise the object
    def __init__(self, k1, k2, k3, k4, k5, k6, d, truncation):
        super(REM, self).__init__()
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.k4 = k4
        self.k5 = k5
        self.k6 = k6
        self.d = d
        self.truncation = truncation
    
    def get_sinusoid(self, L, theta):
        # M = Hadamard_product(L,theta)
        M = L * theta
        s1 = torch.cos(M[:self.k2, ])
        s2 = torch.sin(M[self.k2:(self.k2+self.k3), ])
        s3 = torch.cos(M[(self.k2+self.k3):(self.k2+self.k3+self.k4), ])
        s4 = torch.sin(M[(self.k2+self.k3+self.k4):, ])
        #! numpy? torch?
        s = torch.concat([s1,s2,s3,s4])
        return s
		
    def forward(self, eta, nu, theta):
        # eta -> lamda & nu -> gamma for the initialization?
        lmda = torch.tanh(eta)
        gamma = torch.sigmoid(nu)
        L = self.create_Toeplitz_3D(self.d, self.truncation) # L is of shape (n_heads x query_len x key_len)
        L = torch.from_numpy(L) #?
        # s1,s2,s3,s4
        s = self.get_sinusoid(L, theta)
        print(f's: {s}, type: {type(s)}')
        # powered_lambda = pow(lam, L)
        powered_lambda = lmda ** L
        # powered_gamma = pow(gamma, L)
        powered_gamma = gamma ** L
        # REM = concat(powered_lambda,Hadamard_product(powered_gamma,s))
        rem = torch.concat([powered_lambda, powered_gamma * s])
        return rem
    
    # ! not sure about this
    #! ignore truncation for now??
    # TODO fix
    def create_Toeplitz_3D(self, row, truncation):
        column = row #!
        A = toeplitz(c=column, r=row)
        return A