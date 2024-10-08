import numpy as np

class MultiLayerPerceptron:

    """
    A class representing a simple multilayer perceptron neural network.
    This class initializes the network with predefined weights and biases.

    
    Attributes:
    Network (dict): A dictionary containing the network's weights and biases.
        The keys are 'w1', 'b1', 'w2', 'b2', 'w3', 'b3' for the layers.

        
    Methods:
        sigmoid(s): Applies the sigmoid activation function to an input.
        identity_function(s): Returns the input as is (identity function).
        forward(x): Performs forward propagation through the network given an input.
    
    Example usage:
    >>> mlp = MultilayerPerceptron()
    >>> input = np.array([0.12345,0.31423])
    >>> output = mlp.forward(input)
    >>> print(output)
    
    """

    def __init__(self):
        self.network={}
        self.network['w1'] = np.array([[0.1,0.2,0.3],[0.4,0.5,0.6]])
        self.network['b1'] = np.array([0.1,0.3,0.5])
        self.network['w2'] = np.array([[0.1,0.3],[0.5,0.7],[0.8,0.9]])
        self.network['b2'] = np.array([0.2,0.4])
        self.network['w3'] = np.array([[0.6,0.7],[0.314,0.628]])
        self.network['b3'] = np.array([0.13579,0.2468])

    def sigmoid(self, s):
        return 1/(1+np.exp(-s))
    
    def identityFunction(self, s):
        return s
    
    def forward(self,x):
        w1,w2,w3 = self.network['w1'],self.network['w2'],self.network['w3']
        b1,b2,b3 = self.network['b1'],self.network['b2'],self.network['b3']

        a1 = np.dot(x,w1)+b1
        z1 = self.sigmoid(a1)

        a2 = np.dot(z1,w2)+b2
        z2 = self.sigmoid(a2)

        a3 = np.dot(z2,w3)+b3
        y = self.identityFunction(a3)
        return y

if __name__ == "__main__":

    print("You are in MultilayerPerceptron class!!!")
    print(MultiLayerPerceptron.__doc__)
