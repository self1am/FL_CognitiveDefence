import numpy as np

def update_test():
    dim = 100
    lr = 0.01
    w = np.random.randn(dim, dim) * 0.01
    for i in range(10):
        g_in = np.random.randn(dim) * 10
        g_out = np.random.randn(dim) * 10
            
        pred = w @ g_in
        err = pred - g_out
        
        grad_w = np.outer(err, g_in)
        
        # apply gradient clipping
        grad_norm = np.linalg.norm(grad_w)
        max_norm = 1.0
        if grad_norm > max_norm:
            grad_w = grad_w * (max_norm / grad_norm)
            
        w -= lr * grad_w
        print("w norm:", np.linalg.norm(w))

update_test()
