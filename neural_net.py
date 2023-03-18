"""Neural network model."""

from typing import Sequence

import numpy as np


class NeuralNetwork:
    """A multi-layer fully-connected neural network. The net has an input
    dimension of N, a hidden layer dimension of H, and output dimension C. 
    We train the network with a MLE loss function. The network uses a ReLU
    nonlinearity after each fully connected layer except for the last. 
    The outputs of the last fully-connected layer are passed through
    a sigmoid. 
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        num_layers: int,
        learning_rate: float,
        optimizer: str
    ):
        """Initialize the model. Weights are initialized to small random values
        and biases are initialized to zero. Weights and biases are stored in
        the variable self.params, which is a dictionary with the following
        keys:
        W1: 1st layer weights; has shape (D, H_1)
        b1: 1st layer biases; has shape (H_1,)
        ...
        Wk: kth layer weights; has shape (H_{k-1}, C)
        bk: kth layer biases; has shape (C,)
        Parameters:
            input_size: The dimension D of the input data
            hidden_size: List [H1,..., Hk] with the number of neurons Hi in the
                hidden layer i
            output_size: output dimension C
            num_layers: Number of fully connected layers in the neural network
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.optim = optimizer
        
        assert self.optim == "Adam" or "SGD"
        
        assert len(hidden_sizes) == (num_layers - 1)
        sizes = [input_size] + hidden_sizes + [output_size]

        self.params = {}
        for i in range(1, num_layers + 1):
            self.params["W" + str(i)] = np.random.randn(
                sizes[i - 1], sizes[i]
            ) / np.sqrt(sizes[i - 1])
            self.params["b" + str(i)] = np.zeros(sizes[i])
            
            if self.optim == "Adam":
                self.params["m_w" + str(i)] = np.zeros((sizes[i-1], sizes[i])) # init m, vs
                self.params["m_b" + str(i)] = np.zeros(sizes[i])
                
                self.params["v_w" + str(i)] = np.zeros((sizes[i-1], sizes[i]))
                self.params["v_b" + str(i)] = np.zeros(sizes[i])
                self.t = 0
            
            
    def linear(self, W: np.ndarray, X: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fully connected (linear) layer.
        Parameters:
            W: the weight matrix
            X: the input data
            b: the bias
        Returns:
            the output
        """
        # TODO: implement me
        return (W.T @ X.T).T + b #WX+b
    
    def relu(self, X: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output
        """
        # TODO: implement me
        return np.maximum(0, X)

    def relu_grad(self, X: np.ndarray) -> np.ndarray:
        """Gradient of Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output data
        """
        # TODO: implement me
        X[X<0] = 0 #Relu'(x) = 0 if x<0, 1 if x>0
        X[X>0] = 1
        return X

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
      # TODO ensure that this is numerically stable
        y  = np.zeros(np.shape(x))
        y[x > -100] =  1/(1+np.exp(-x[x > -100]))  #exp(100) is pretty big, so we'll just set y=0 if x>100     
        return y
        
    def mse(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        # TODO implement this
        return 1/2 * np.mean((y-p)**2)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute the outputs for all of the data samples.
        Hint: this function is also used for prediction.
        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training or
                testing sample
        Returns:
            Matrix of shape (N, C) 
        """
        self.outputs = {}        
        # TODO: implement me. You'll want to store the output of each layer in
        # self.outputs as it will be used during back-propagation. You can use
        # the same keys as self.params. You can use functions like
        # self.linear, self.relu, and self.mse in here.
        self.outputs[0] = X # outputs[0] = inputs
        for i in range(1,self.num_layers+1):
            W = self.params["W" + str(i)]
            b = self.params["b" + str(i)]
            X = self.linear(W, X, b) # linear layer
            
            
            if i == self.num_layers: # nonlinearity
                X = self.sigmoid(X) # last layer sigmoid
            else:
                X = self.relu(X) # hidden layers = ReLU
                
            self.outputs[i] = X # put output in
            
        return X # last output = output of net
    
    def backward(self, y: np.ndarray) -> float:
        """Perform back-propagation and compute the gradients and losses.
        Parameters:
            y: training value targets
        Returns:
            Total loss for this batch of training samples
        """
        self.gradients = {}
        # TODO: implement me. You'll want to store the gradient of each
        # parameter in self.gradients as it will be used when updating each
        # parameter and during numerical gradient checks. You can use the same
        # keys as self.params. You can add functions like self.linear_grad,
        # self.relu_grad, and self.softmax_grad if it helps organize your code.
 
        upstream_grad = -(y - self.outputs[self.num_layers]) / (y.shape[0] * y.shape[1]) # dL/dh_N 

        for i in range(self.num_layers, 0, -1):
            local_grad_w = 1.0 # reset local gradients
            local_grad_b = 1.0
            
            if i == self.num_layers: # h_N = sigmoid(w_N @ h_(N-1) + b)
                
                siggrad = self.outputs[i] * (1-self.outputs[i])

                grad_w = (upstream_grad * siggrad).T @ self.outputs[i-1] # upstream grad * sigmoid' @ h_{N-1}
                self.gradients["W" + str(i)] = grad_w.T # store gradients for update

                grad_b = (upstream_grad * siggrad).T @ np.ones(y.shape[0]) # upstream * sigmoid' * 1
                self.gradients["b" + str(i)] = grad_b
               
                upstream_grad = (upstream_grad * siggrad) @ self.params["W" + str(i)].T # (dH_N/dH_{N-1}) = sig' * w_N
                
            else: # h_k = ReLU(w_k @ h_(k-1) + b)
                
                inputs = self.outputs[i-1]
                
                relugrad = self.relu_grad(self.outputs[i])
                
                grad_w = (upstream_grad * relugrad).T @ self.outputs[i-1] # upstream grad * ReLU'(h_N) @ h_{N-1}
                
                grad_b = (upstream_grad * relugrad).T @ np.ones(self.outputs[i].shape[0]) # upstream * ReLU'(h_N) * 1
                
                self.gradients["W" + str(i)] = grad_w.T
                self.gradients["b" + str(i)] = grad_b

                upstream_grad = (upstream_grad * relugrad) @ self.params["W" + str(i)].T # dh_k/dh_{k-1} = relu' * w_k
             
        return self.mse(y, self.outputs[self.num_layers]) # loss

    def update(
        self,
        lr: float = None,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        opt: str = None
    ):
        """Update the parameters of the model using the previously calculated
        gradients.
        Parameters:
            lr: Learning rate
            b1: beta 1 parameter (for Adam)
            b2: beta 2 parameter (for Adam)
            eps: epsilon to prevent division by zero (for Adam)
            opt: optimizer, either 'SGD' or 'Adam'
        """
        # TODO: implement me. You'll want to add an if-statement that can
        # handle updates for both SGD and Adam depending on the value of opt.
        if opt == None:
            opt = self.optim
            
        if lr == None:
            lr = self.learning_rate
        
        if opt == "SGD":
            for i in range(1,self.num_layers+1):
                self.params["W" + str(i)] -= lr * self.gradients["W" + str(i)]
                self.params["b" + str(i)] -= lr * self.gradients["b" + str(i)]
        else: # Adam
            for i in range(1,self.num_layers+1):
                self.t += 1
                self.params["m_w" + str(i)] = b1 * self.params["m_w" + str(i)] + (1-b1) * self.gradients["W" + str(i)] # update m
                self.params["m_b" + str(i)] = b1 * self.params["m_b" + str(i)] + (1-b1) * self.gradients["b" + str(i)]
                
                self.params["v_w" + str(i)] = b2 * self.params["v_w" + str(i)] + (1-b2) * self.gradients["W" + str(i)]**2 # update v
                self.params["v_b" + str(i)] = b2 * self.params["v_b" + str(i)] + (1-b2) * self.gradients["b" + str(i)]**2 # update v
                
                m_hat_w = self.params["m_w" + str(i)] / (1-b1**self.t) # bias correction
                m_hat_b = self.params["m_b" + str(i)] / (1-b1**self.t)
                
                v_hat_w = self.params["v_w" + str(i)] / (1-b2**self.t)
                v_hat_b = self.params["v_b" + str(i)] / (1-b2**self.t)
                 
                self.params["W" + str(i)] -= lr / (np.sqrt(v_hat_w) + eps) * m_hat_w # update weights
                self.params["b" + str(i)] -= lr / (np.sqrt(v_hat_b) + eps) * m_hat_b
            