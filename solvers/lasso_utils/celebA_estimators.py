"""Estimators for compressed sensing"""
# pylint: disable = C0301, C0103, C0111, R0914

import copy
import heapq
import numpy as np
from . import utils
#import utils
import scipy.fftpack as fftpack
import pywt


def dct2(image_channel):
    return fftpack.dct(fftpack.dct(image_channel.T, norm='ortho').T, norm='ortho')


def idct2(image_channel):
    return fftpack.idct(fftpack.idct(image_channel.T, norm='ortho').T, norm='ortho')


def vec(channels):
    image = np.zeros((64, 64, 3))
    for i, channel in enumerate(channels):
        image[:, :, i] = channel
    return image.reshape([-1])


def devec(vector):
    image = np.reshape(vector, [64, 64, 3])
    channels = [image[:, :, i] for i in range(3)]
    return channels


def wavelet_basis(path='./solvers/lasso_utils/wavelet_basis.npy'):
    W_ = np.load(path)
    # W_ initially has shape (4096,64,64), i.e. 4096 64x64 images
    # reshape this into 4096x4096, where each row is an image
    # take transpose to make columns images
    W_ = W_.reshape((4096, 4096))
    W = np.zeros((12288, 12288))
    W[0::3, 0::3] = W_
    W[1::3, 1::3] = W_
    W[2::3, 2::3] = W_
    return W


def lasso_dct_estimator(hparams):  #pylint: disable = W0613
    """LASSO with DCT"""
    def estimator(A_val, y_batch_val, hparams):
        # One can prove that taking 2D DCT of each row of A,
        # then solving usual LASSO, and finally taking 2D ICT gives the correct answer.
        A_new = copy.deepcopy(A_val)
        for i in range(A_val.shape[1]):
            A_new[:, i] = vec([dct2(channel) for channel in devec(A_new[:, i])])

        x_hat_batch = []
        for j in range(hparams.batch_size):
            y_val = y_batch_val[j]
            z_hat = utils.solve_lasso(A_new, y_val, hparams)
            x_hat = vec([idct2(channel) for channel in devec(z_hat)]).T
            x_hat = np.maximum(np.minimum(x_hat, 1), -1)
            x_hat_batch.append(x_hat)
        return x_hat_batch
    return estimator


def lasso_wavelet_estimator(hparams):  #pylint: disable = W0613
    """LASSO with Wavelet"""
    def estimator(A_val, y_batch_val, hparams):
        x_hat_batch = []
        W = wavelet_basis()
        WA = np.dot(W, A_val)
        for j in range(hparams.batch_size):
            y_val = y_batch_val[j]
            z_hat = utils.solve_lasso(WA, y_val, hparams)
            x_hat = np.dot(z_hat, W)
            x_hat_max = np.abs(x_hat).max()
            x_hat = x_hat / (1.0 * x_hat_max)
            x_hat_batch.append(x_hat)
        x_hat_batch = np.asarray(x_hat_batch)
        return x_hat_batch
    return estimator


def lasso_wavelet_ycbcr_estimator(hparams):  #pylint: disable = W0613
    """LASSO with Wavelet in YCbCr"""

    def estimator(A_val, y_batch_val, hparams):
        x_hat_batch = []

        W = wavelet_basis()
        # U, V = utils.RGB_matrix()
        # V = (V/127.5) - 1.0
        # U = U/127.5
        def convert(W):
            # convert W from YCbCr to RGB
            W_ = W.copy()
            V = np.zeros((12288, 1))
            # R
            V[0::3] = ((255.0/219.0)*(-16.0)) + ((255.0*0.701/112.0)*(-128.0))
            W_[:, 0::3] = (255.0/219.0)*W[:, 0::3] + (0.0)*W[:, 1::3] + (255.0*0.701/112.0)*W[:, 2::3]
            # G
            V[1::3] = ((255.0/219.0)*(-16.0)) - ((0.886*0.114*255.0/(112.0*0.587)) *(-128.0)) - ((255.0*0.701*0.299/(112.0*0.587))*(-128.0))
            W_[:, 1::3] = (255.0/219.0)*W[:, 0::3] - (0.886*0.114*255.0/(112.0*0.587))*W[:, 1::3] - (255.0*0.701*0.299/(112.0*0.587))*W[:, 2::3]
            # B
            V[2::3] = ((255.0/219.0)*(-16.0)) + ((0.886*255.0/(112.0))*(-128.0))
            W_[:, 2::3] = (255.0/219.0)*W[:, 0::3]  + (0.886*255.0/(112.0))*W[:, 1::3] + 0.0*W[:, 2::3]
            return W_, V

        # WU = np.dot(W, U.T)
        WU, V = convert(W)
        WU = WU/127.5
        V = (V/127.5) - 1.0
        WA = np.dot(WU, A_val)
        y_batch_val_temp = y_batch_val - np.dot(V.T, A_val)
        for j in range(hparams.batch_size):
            y_val = y_batch_val_temp[j]
            z_hat = utils.solve_lasso(WA, y_val, hparams)
            x_hat = np.dot(z_hat, WU) + V.ravel()
            x_hat_max = np.abs(x_hat).max()
            x_hat = x_hat / (1.0 * x_hat_max)
            x_hat_batch.append(x_hat)
        x_hat_batch = np.asarray(x_hat_batch)
        return x_hat_batch

    return estimator


def k_sparse_wavelet_estimator(hparams): #pylint: disable = W0613
    """Best k-sparse wavelet projector"""
    def estimator(A_val, y_batch_val, hparams): #pylint: disable = W0613
        if hparams.measurement_type != 'project':
            raise RuntimeError
        y_batch_val /= np.sqrt(hparams.n_input)
        x_hat_batch = []
        for y_val in y_batch_val:
            y_val_reshaped = np.reshape(y_val, [64, 64, 3])
            x_hat_reshaped = k_sparse_reconstr(y_val_reshaped, hparams.sparsity)
            x_hat_flat = np.reshape(x_hat_reshaped, [-1])
            x_hat_batch.append(x_hat_flat)
        x_hat_batch = np.asarray(x_hat_batch)
        x_hat_batch = np.maximum(np.minimum(x_hat_batch, 1), -1)
        return x_hat_batch
    return estimator


def get_wavelet(x):
    coefs_list = []
    for i in range(3):
        coefs_list.append(pywt.wavedec2(x[:, :, i], 'db1'))
    return coefs_list


def get_image(coefs_list):
    x = np.zeros((64, 64, 3))
    for i in range(3):
        x[:, :, i] = pywt.waverec2(coefs_list[i], 'db1')
    return x


def get_heap(coefs_list):
    heap = []
    for t, coefs in enumerate(coefs_list):
        for i, a in enumerate(coefs):
            for j, b in enumerate(a):
                for m, c in enumerate(b):
                    try:
                        for n, val in enumerate(c):
                            heapq.heappush(heap, (-abs(val), [t, i, j, m, n, val]))
                    except:
                        val = c
                        heapq.heappush(heap, (-abs(val), [t, i, j, m, val]))
    return heap


def k_sparse_reconstr(x, k):
    coefs_list = get_wavelet(x)
    heap = get_heap(coefs_list)

    y = 0*x
    coefs_list_sparse = get_wavelet(y)
    for i in range(k):
        _, idxs_val = heapq.heappop(heap)
        if len(idxs_val) == 5:
            t, i, j, m, val = idxs_val
            coefs_list_sparse[t][i][j][m] = val
        else:
            t, i, j, m, n, val = idxs_val
            coefs_list_sparse[t][i][j][m][n] = val
    x_sparse = get_image(coefs_list_sparse)
    return x_sparse
