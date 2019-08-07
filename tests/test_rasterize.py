import scipy.misc
import unittest

import chainer
import chainer.functions as cf
import chainer.testing
import numpy as np
import cupy as cp
import imageio

import neural_renderer


class Parameter(chainer.Link):
    def __init__(self, x):
        super(Parameter, self).__init__()
        with self.init_scope():
            self.x = chainer.Parameter(x)

    def __call__(self):
        return self.x


class TestRasterize(unittest.TestCase):
    def stest_forward_case1(self):
        # load reference image by blender
        ref = imageio.imread('./tests/data/teapot_blender.png')
        ref = (ref.min(-1) != 255).astype('float32')
        ref = chainer.cuda.to_gpu(ref)

        target_num = 2
        vertices, faces = neural_renderer.load_obj('./tests/data/teapot.obj')
        vertices_batch = np.tile(vertices[None, :, :], (4, 1, 1)) * 0
        vertices_batch[target_num] = vertices
        vertices, faces = neural_renderer.to_gpu((vertices_batch, faces))

        renderer = neural_renderer.Renderer()
        renderer.anti_aliasing = False
        renderer.viewpoints = neural_renderer.get_points_from_angles(2.732, 0, 0)
        images = renderer.render_silhouettes(vertices, faces).data[target_num]

        chainer.testing.assert_allclose(images, ref, atol=2e-3)

    def stest_forward_case2(self):
        data = [
            [
                './tests/data/4e49873292196f02574b5684eaec43e9/model.obj',
                neural_renderer.get_points_from_angles(2.5, 10, -90),
                './tests/data/4e49873292196f02574b5684eaec43e9.png',
            ],
            [
                './tests/data/1cde62b063e14777c9152a706245d48/model.obj',
                neural_renderer.get_points_from_angles(2.5, 10, 60),
                './tests/data/1cde62b063e14777c9152a706245d48.png',
            ]
        ]

        renderer = neural_renderer.Renderer()
        renderer.draw_backside = False
        for i, (filename, viewpoint, reference) in enumerate(data):
            renderer.viewpoints = viewpoint
            ref = neural_renderer.imread(reference)

            vertices, faces, vertices_t, faces_t, textures = neural_renderer.load_obj(filename, load_textures=True)
            vertices, faces, vertices_t, faces_t, textures = neural_renderer.to_gpu(
                (vertices[None, :, :], faces, vertices_t[None, :, :], faces_t, textures[None, :, :, :]))

            images = renderer.render(vertices, faces, vertices_t, faces_t, textures).data
            image = images[0].transpose((1, 2, 0))
            # imageio.toimage(image.get(), cmin=0, cmax=1).save(reference)

            chainer.testing.assert_allclose(ref, image, atol=1e-2)

    def stest_forward_case3(self):
        # load reference image by blender
        ref = imageio.imread('./tests/data/teapot_depth.png')
        ref = ref.astype('float32') / 255.
        ref = chainer.cuda.to_gpu(ref)

        target_num = 2
        vertices, faces = neural_renderer.load_obj('./tests/data/teapot.obj')
        vertices_batch = np.tile(vertices[None, :, :], (4, 1, 1)) * 0
        vertices_batch[target_num] = vertices
        vertices, faces = neural_renderer.to_gpu((vertices_batch, faces))

        renderer = neural_renderer.Renderer()
        renderer.anti_aliasing = False
        renderer.viewpoints = neural_renderer.get_points_from_angles(2, 30., 0)
        images = renderer.render_depth(vertices, faces).data[target_num]
        images = (images - images.min()) / (images.max() - images.min())
        # imageio.toimage(images.get()).save('./tests/data/teapot_depth.png')

        chainer.testing.assert_allclose(images, ref, atol=2e-3)

    def test_forward_case4(self):
        # lights
        ref = imageio.imread('./tests/data/teapot_blender.png')
        ref = (ref.min(-1) != 255).astype('float32')
        ref = chainer.cuda.to_gpu(ref)

        target_num = 2
        vertices, faces = neural_renderer.load_obj('./tests/data/teapot.obj')
        vertices_batch = np.tile(vertices[None, :, :], (4, 1, 1)) * 0
        vertices_batch[target_num] = vertices
        vertices, faces = neural_renderer.to_gpu((vertices_batch, faces))
        vertices_t, faces_t, textures = neural_renderer.create_textures(faces.shape[0])
        vertices_t = cf.tile(vertices_t[None, :, :], (4, 1, 1)).data
        textures = cf.tile(textures[None, :, :, :], (4, 1, 1, 1)).data
        vertices_t = chainer.cuda.to_gpu(vertices_t)
        faces_t = chainer.cuda.to_gpu(faces_t)
        textures = chainer.cuda.to_gpu(textures)

        lights = []
        light_color = cp.random.uniform(0., 1., size=(4, 3)).astype('float32') * 0.5
        light_direction = cf.normalize(cp.random.uniform(0., 1., size=(4, 3)).astype('float32'))
        lights.append(neural_renderer.DirectionalLight(light_color, light_direction))
        light_color = cp.random.uniform(0., 1., size=(4, 3)).astype('float32') * 0.5
        lights.append(neural_renderer.AmbientLight(light_color))
        light_color = cp.random.uniform(0., 1., size=(4, 3)).astype('float32') * 0.5
        lights.append(neural_renderer.SpecularLight(light_color))
        renderer = neural_renderer.Renderer()
        renderer.viewpoints = neural_renderer.get_points_from_angles(2.732, 30, 30)
        renderer.draw_backside = False
        images = renderer.render_rgb(vertices, faces, vertices_t, faces_t, textures, lights=lights).data[target_num]

        import pylab
        pylab.imshow(images.get().transpose((1, 2, 0))[:, :, :3])
        pylab.show()

    def stest_backward_case1(self):
        vertices = [
            [0.1, 0.1, 1.],
            [-0.1, 0.1, 1.],
            [-0.1, -0.1, 1.],
            [0.1, -0.1, 1.],
        ]
        faces = [[0, 1, 2], [0, 2, 3]]
        # vertices = [
        #     [0.3, 0.4, 1.],
        #     [-0.3, 0.6, 1.],
        #     [-0.3, 0.62, 1.],
        # ]
        # faces = [[0, 1, 2]]

        ref = neural_renderer.imread('./tests/data/gradient.png')
        ref = 1 - ref
        ref = ref[:, :, 0]
        ref = chainer.cuda.to_gpu(ref)

        vertices = np.array(vertices, 'float32')
        faces = np.array(faces, 'int32')
        vertices, faces, ref = neural_renderer.to_gpu((vertices, faces, ref))
        vertices = Parameter(vertices)
        optimizer = chainer.optimizers.Adam(0.003, beta1=0.5)
        optimizer.setup(vertices)

        for i in range(350):
            images = neural_renderer.rasterize_silhouettes(
                vertices()[None, :, :], faces, image_size=256, anti_aliasing=False)
            image = images[0]

            iou = cf.sum(image * ref) / cf.sum(image + ref - image * ref)
            iou = 1 - iou
            loss = iou

            optimizer.target.cleargrads()
            loss.backward()
            optimizer.update()

            # imageio.toimage(image.data.get()).save('../tmp/t%d.png' % i)
            # print i, loss.data, iou.data

            if float(iou.data) < 0.01:
                return
        raise Exception


if __name__ == '__main__':
    unittest.main()
