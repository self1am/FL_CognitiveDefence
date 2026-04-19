import numpy as np

def update_test():
    dim = 100
    lr = 0.01
    w = np.random.randn(dim, dim) * 0.01
    for i in range(10):
        g_in = np.random.randn(dim) * 10
        g_out = np.random.randn(dim) * 10
        
        # normalize
        in_norm = np.linalg.norm(g_in)
        if in_norm > 1e-5:
            g_in = g_in / in_norm
            
        pred = w @ g_in
        err = pred - g_out
        
        grad_w = np.outer(err, g_in)
        w -= lr * grad_w
        print("w norm:", np.linalg.norm(w))

update_test()
