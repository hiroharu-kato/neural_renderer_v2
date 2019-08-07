"""
Example 3. Optimizing textures.
"""
import argparse
import glob
import os
import subprocess

import chainer
import chainer.functions as cf
import numpy as np
import scipy.misc
import tqdm

import neural_renderer


class Model(chainer.Link):
    def __init__(self, filename_obj, filename_ref):
        super(Model, self).__init__()

        with self.init_scope():
            # load .obj
            vertices, faces = neural_renderer.load_obj(filename_obj)
            self.vertices = vertices[None, :, :]
            self.faces = faces

            # create textures
            vertices_t, faces_t, textures = neural_renderer.create_textures(self.faces.shape[0], texture_size=4)
            self.vertices_t = vertices_t[None, :, :]
            self.faces_t = faces_t
            self.textures = chainer.Parameter(textures[None, :, :, :])

            # load reference image
            self.image_ref = neural_renderer.imread(filename_ref)

            # setup renderer
            renderer = neural_renderer.Renderer()
            renderer.perspective = False
            self.renderer = renderer

    def to_gpu(self, device=None):
        super(Model, self).to_gpu(device)
        self.faces = chainer.cuda.to_gpu(self.faces, device)
        self.vertices = chainer.cuda.to_gpu(self.vertices, device)
        self.vertices_t = chainer.cuda.to_gpu(self.vertices_t, device)
        self.faces_t = chainer.cuda.to_gpu(self.faces_t, device)
        self.image_ref = chainer.cuda.to_gpu(self.image_ref, device)

    def __call__(self):
        self.renderer.viewpoints = neural_renderer.get_points_from_angles(2.732, 0, np.random.uniform(0, 360))
        images = self.renderer.render_rgb(
            self.vertices, self.faces, self.vertices_t, self.faces_t, cf.tanh(self.textures))
        loss = cf.sum(cf.square(images[0] - self.image_ref.transpose((2, 0, 1))))
        return loss


def make_gif(working_directory, filename):
    # generate gif (need ImageMagick)
    options = '-delay 8 -loop 0 -layers optimize'
    subprocess.call('convert %s %s/_tmp_*.png %s' % (options, working_directory, filename), shell=True)
    for filename in glob.glob('%s/_tmp_*.png' % working_directory):
        os.remove(filename)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('-io', '--filename_obj', type=str, default='./examples/data/teapot.obj')
    parser.add_argument('-ir', '--filename_ref', type=str, default='./examples/data/example3_ref.png')
    parser.add_argument('-or', '--filename_output', type=str, default='./examples/data/example3_result.gif')
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()
    working_directory = os.path.dirname(args.filename_output)

    model = Model(args.filename_obj, args.filename_ref)
    model.to_gpu()

    optimizer = chainer.optimizers.Adam(alpha=0.01)
    optimizer.setup(model)
    loop = tqdm.tqdm(range(300))
    for _ in loop:
        loop.set_description('Optimizing')
        optimizer.target.cleargrads()
        loss = model()
        loss.backward()
        optimizer.update()

    # draw object
    loop = tqdm.tqdm(range(0, 360, 4))
    for num, azimuth in enumerate(loop):
        loop.set_description('Drawing')
        model.renderer.viewpoints = neural_renderer.get_points_from_angles(2.732, 0, azimuth)
        images = model.renderer.render_rgb(
            model.vertices, model.faces, model.vertices_t, model.faces_t, cf.tanh(model.textures))
        image = images.data.get()[0].transpose((1, 2, 0))
        scipy.misc.toimage(image, cmin=0, cmax=1).save('%s/_tmp_%04d.png' % (working_directory, num))
    make_gif(working_directory, args.filename_output)



if __name__ == '__main__':
    run()
