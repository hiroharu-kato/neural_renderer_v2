import unittest

import chainer
import chainer.functions as cf
import chainer.gradient_check
import chainer.testing
import cupy as cp
import numpy as np

import neural_renderer


class TestDifferentiation(unittest.TestCase):
    def test_backward(self):
        images = np.random.normal(size=(10, 32, 32, 3)).astype('float32')
        x = np.tile(np.arange(32).astype('float32')[None, None, :, None], (10, 32, 1, 1))
        y = np.tile(np.arange(32).astype('float32')[None, :, None, None], (10, 1, 32, 1))
        coordinates = np.concatenate((x, y), axis=-1)
        coordinates = ((coordinates / 31) * 2 - 1) * 31. / 32.
        noise = np.random.normal(size=(10, 32, 32, 3)).astype('float32')
        step = 2 / 32.

        images = chainer.cuda.to_gpu(images)
        coordinates = chainer.Variable(chainer.cuda.to_gpu(coordinates))
        noise = chainer.cuda.to_gpu(noise)

        loss = cf.sum(neural_renderer.differentiation(images, coordinates) * noise)
        loss.backward()

        grad_coordinates = coordinates.grad

        for i in range(100):
            yi = np.random.randint(1, 31)
            xi = np.random.randint(1, 31)

            images_yb = images.copy()
            images_yb[:, yi - 1, xi] = images[:, yi, xi].copy()
            images_yb[:, yi, xi] = images[:, yi + 1, xi].copy()
            grad_yb = ((images_yb - images) * noise).sum((1, 2, 3)) / step
            grad_yb = cp.minimum(grad_yb, cp.zeros_like(grad_yb))

            images_yt = images.copy()
            images_yt[:, yi + 1, xi] = images[:, yi, xi].copy()
            images_yt[:, yi, xi] = images[:, yi - 1, xi].copy()
            grad_yt = ((images_yt - images) * noise).sum((1, 2, 3)) / step
            grad_yt = cp.minimum(grad_yt, cp.zeros_like(grad_yt))

            grad_y_abs = cp.maximum(cp.abs(grad_yb), cp.abs(grad_yt))

            chainer.testing.assert_allclose(grad_y_abs, cp.abs(grad_coordinates[:, yi, xi, 1]))

            images_xl = images.copy()
            images_xl[:, yi, xi - 1] = images[:, yi, xi].copy()
            images_xl[:, yi, xi] = images[:, yi, xi + 1].copy()
            grad_xl = ((images_xl - images) * noise).sum((1, 2, 3)) / step
            grad_xl = cp.minimum(grad_xl, cp.zeros_like(grad_xl))

            images_xr = images.copy()
            images_xr[:, yi, xi + 1] = images[:, yi, xi].copy()
            images_xr[:, yi, xi] = images[:, yi, xi - 1].copy()
            grad_xr = ((images_xr - images) * noise).sum((1, 2, 3)) / step
            grad_xr = cp.minimum(grad_xr, cp.zeros_like(grad_xr))

            grad_x_abs = cp.maximum(cp.abs(grad_xl), cp.abs(grad_xr))

            chainer.testing.assert_allclose(grad_x_abs, cp.abs(grad_coordinates[:, yi, xi, 0]))


if __name__ == '__main__':
    unittest.main()
