#python3 Steven 10/02/20
#pytorch study
import numpy as np
import torch

def test():
    x = torch.tensor([3,5]) #torch.tensor([4],dtype=torch.float)
    y = torch.tensor([2,1])
    print(x*y)
    x = torch.zeros([2,5])
    print(x.shape)
    
    y = torch.rand([2,3])
    print(y.shape,y)
    
    y = y.view([3,2]) #reshape
    print(y.shape,y)
    
def numpyBP0():
    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, D_out = 1, 1, 1

    # Create random input and output data
    x = np.array([[5.0]]) #np.random.randn(N, D_in)
    y = np.array([[2.0]]) #np.random.randn(N, D_out)

    print('x=',x)
    print('y=',y)
    # Randomly initialize weights
    w1 = np.random.randn(D_in, D_out)

    learning_rate = 1e-3
    for t in range(500):
        # Forward pass: compute predicted y
        h = x.dot(w1)
        h_relu = np.maximum(h, 0)
        y_pred = h_relu

        # Compute and print loss
        loss = np.square(y_pred - y).sum()
        if t%20 == 0:
            print(t, loss)

        # Backprop to compute gradients of w1 and w2 with respect to loss
        grad_y_pred = 2.0 * (y_pred - y)
        grad_w1 = x.T.dot(grad_y_pred)
       
        # Update weights
        w1 -= learning_rate * grad_w1
        
    print('w1=',w1)

def numpyBP0_1():
    N, D_in, D_out = 1, 1, 1

    # Create random input and output data
    x = np.array([[5.0]]) #np.random.randn(N, D_in)
    y = np.array([[2.0]]) #np.random.randn(N, D_out)

    print('x=',x)
    print('y=',y)
    # Randomly initialize weights
    w1 = np.random.randn(D_in, D_out)
    b1 = np.random.randn(D_in, D_out)

    learning_rate = 1e-3
    for t in range(500):
        # Forward pass: compute predicted y
        h = x.dot(w1) + b1
        h_relu = np.maximum(h, 0)
        y_pred = h_relu

        # Compute and print loss
        loss = np.square(y_pred - y).sum()
        if t%20 == 0:
            print(t, loss)

        # Backprop to compute gradients of w1 and w2 with respect to loss
        grad_y_pred = 2.0 * (y_pred - y)
        grad_w1 = x.T.dot(grad_y_pred)
       
        # Update weights
        w1 -= learning_rate * grad_w1
        b1 -= learning_rate * grad_w1
        
    print('w1=',w1)
    print('b1=',b1)

def numpyBP1():
    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H, D_out = 64, 1000, 100, 10

    # Create random input and output data
    x = np.random.randn(N, D_in)
    y = np.random.randn(N, D_out)

    # Randomly initialize weights
    w1 = np.random.randn(D_in, H)
    w2 = np.random.randn(H, D_out)

    learning_rate = 1e-6
    for t in range(100):
        # Forward pass: compute predicted y
        h = x.dot(w1)
        h_relu = np.maximum(h, 0)
        y_pred = h_relu.dot(w2)

        # Compute and print loss
        loss = np.square(y_pred - y).sum()
        print(t, loss)

        # Backprop to compute gradients of w1 and w2 with respect to loss
        grad_y_pred = 2.0 * (y_pred - y)
        grad_w2 = h_relu.T.dot(grad_y_pred)
        grad_h_relu = grad_y_pred.dot(w2.T)
        grad_h = grad_h_relu.copy()
        grad_h[h < 0] = 0
        grad_w1 = x.T.dot(grad_h)

        # Update weights
        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2
    
def torchBP():
    dtype = torch.float
    device = torch.device("cpu")
    # device = torch.device("cuda:0") # Uncomment this to run on GPU

    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H, D_out = 64, 1000, 100, 10

    # Create random input and output data
    x = torch.randn(N, D_in, device=device, dtype=dtype)
    y = torch.randn(N, D_out, device=device, dtype=dtype)

    # Randomly initialize weights
    w1 = torch.randn(D_in, H, device=device, dtype=dtype)
    w2 = torch.randn(H, D_out, device=device, dtype=dtype)

    learning_rate = 1e-6
    for t in range(500):
        # Forward pass: compute predicted y
        h = x.mm(w1)
        h_relu = h.clamp(min=0)
        y_pred = h_relu.mm(w2)

        # Compute and print loss
        loss = (y_pred - y).pow(2).sum().item()
        if t % 50 == 0:
            print(t, loss)

        # Backprop to compute gradients of w1 and w2 with respect to loss
        grad_y_pred = 2.0 * (y_pred - y)
        grad_w2 = h_relu.t().mm(grad_y_pred)
        grad_h_relu = grad_y_pred.mm(w2.t())
        grad_h = grad_h_relu.clone()
        grad_h[h < 0] = 0
        grad_w1 = x.t().mm(grad_h)

        # Update weights using gradient descent
        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2
    
def main():
    test()
    #numpyBP0()
    #numpyBP0_1()
    #numpyBP1()
    #torchBP()
    
if __name__ == '__main__':
    main()
    