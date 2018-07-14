import numpy as np
import h5py
# convolution functions: zero padding,convolve window, convolution forward/backward
# pooling functions: pooling forward, create mask, distribute value,pooling backward

# padding lets a layer without shrinking the height and width of volumes,helps us keep more information at the border of an image
# 'same' convolution, the height/width is exactly preserved after one layer


# zero padding
def zero_pad(X, pad):
    X_pad = np.pad(X, ((0,0),(pad,pad),(pad,pad),(0,0)),'constant',constant_values = ((0,0),(0,0),(0,0),(0,0)))
    return X_pad

def conv_single_step(a_slice_prev, W, b):
    s = np.multiply(W, a_slice_prev)
    Z = np.sum(s)
    Z = Z + float(b)

    return Z

# CNN-forward pass
def conv_forward(A_prev, W, b, hparameters):
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    (f, f, n_C_prev, n_C) = W.shape

    stride = hparameters['stride']
    pad = hparameters['pad']

    n_H = int((n_H_prev + 2*pad - f)/stride) + 1
    n_W = int((n_W_prev + 2*pad - f)/stride) + 1

    Z = np.zeros((m,n_H,n_W,n_C))

    A_prev_pad = zero_pad(A_prev, pad)

    for i in range(m):
        a_prev_pad = A_prevv_pad[i,:,:,:]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):

                    vert_start = h*stride
                    vert_end = vert_start + f
                    horiz_start = w*stride
                    horiz_end = horiz_start + f

                    a_slice_prev = a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, W[:,:,:,c], b[:,:,:,c])\
                    # activation function: A[i, h, w, c] = activation(Z[i, h, w, c])

    cache = (A_prev, W, b, hparametres)

    return Z, cache

# pooling layer reduces the height and width of input, reduces computation, makes feature detectors more invariant to the position in the input
# max-pooling, average-pooling

# forward pooling
def pool_forward(A_prev, hparameters, mode='max'):
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    f = hparameters['f']
    stride = hparameters['stride']

    n_H = int((n_H_prev - f)/stride) + 1
    n_W = int((n_W_prev - f)/stride) + 1
    n_C = n_C_prev

    A = np.zeros((m,n_H,n_W.n_C))

    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):

                    vert_start = h*stride
                    vert_end = vert_start + f
                    horiz_start = w*stride
                    horiz_end = horiz_start + f

                    a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]

                    if mode == 'max':
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == 'average':
                        A[i, h, w, c] = np.mean(a_prev_slice)

    cache = (A_prev, hparameters)

    return A, cache

# backpropagation in CNN
def conv_backward(dZ, cache):
    (A_prev, W, b, hparameters) = cache
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f,f,n_C_prev, n_C) = W.shape

    stride = hparameters['stride']
    pad = hparameters['pad']

    (m, n_H, n_W, n_C) = dZ.shape

    dA_prev = np.zeros((m,n_H_prev, n_W_prev, n_C_prev))
    dW = np.zeros((f,f,n_C_prev, n_C))
    db = np.zeros((1,1,1,n_C))

    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)

    for i in range(m):

        a_prev_pad = A_prev_pad[i,:,:,:]
        da_prev_pad = dA_prev_pad[i,:,:,:]

        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):

                    vert_start = h*stride
                    vert_end = vert_start + f
                    horiz_start = w*stride
                    horiz_end = horiz_start + f

                    a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end,:]

                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end,:] += W[:,:,:,c] * dZ[i,h,w,c]
                    dW[:, :, :, c] += np.multiply(a_slice, dZ[i, h, w, c])
                    db[:, :, :, c] += dZ[i, h, w, c]
        dA_prev[i,:,:,:] = da_prev_pad[pad:-pad,pad:-pad,:]

    return dA_prev, dW, db

def create_mask_from_window(x):
    mask = (x == np.max(x))
    return mask

def distribute_value(dz, shape):
    (n_H, n_W) = shape
    average = dz/(n_H*n_W)
    a = np.matrix([[average for i in range(n_H)] for j in range(n_W)])
    return a

def pool_backward(dA, cache, mode='max'):
    (A_prev, hparameters) = cache

    stride = hparameters['stride']
    f = hparameters['f']

    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    m, n_H, n_W, n_C = dA.shape

    dA_prev = np.zeros((m,n_H_prev,n_W_prev,n_C_prev))

    for i in range(m):
        a_prev = A_prev[i,:,:,:]

        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h*stride
                    vert_end = vert_start + f
                    horiz_start = w*stride
                    horiz_end = horiz_start + f

                    if mode == 'max':
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end,c]
                        mask = create_mask_from_window(a_prev_slice)
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += mask*dA[i,h,w,c]
                    elif mode == 'average':
                        da = dA[i,h,w,c]
                        shape = (f,f)
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end,c] += distribute_value(da, shape)

    return dA_prev

