import glob
import os
import unittest

import chainer
import chainer.testing

import neural_renderer


class TestSaveObj(unittest.TestCase):
    def test_case1(self):
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
        filename_tmp = './tests/data/tmp.obj'

        renderer = neural_renderer.Renderer()
        renderer.draw_backside = False
        for i, (filename, viewpoint, reference) in enumerate(data):
            renderer.viewpoints = viewpoint
            ref = neural_renderer.imread(reference)

            vertices, faces, vertices_t, faces_t, textures = neural_renderer.load_obj(filename, load_textures=True)
            neural_renderer.save_obj(filename_tmp, vertices, faces, vertices_t, faces_t, textures)
            vertices, faces, vertices_t, faces_t, textures = neural_renderer.load_obj(filename_tmp, load_textures=True)
            vertices, faces, vertices_t, faces_t, textures = neural_renderer.to_gpu(
                (vertices[None, :, :], faces, vertices_t[None, :, :], faces_t, textures[None, :, :, :]))

            images = renderer.render(vertices, faces, vertices_t, faces_t, textures).data
            image = images[0].transpose((1, 2, 0))

            chainer.testing.assert_allclose(ref, image, atol=1e-2, rtol=1e-2)

        for f in glob.glob('./tests/data/tmp*'):
            os.remove(f)


if __name__ == '__main__':
    unittest.main()
