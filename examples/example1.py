"""
Example 1. Drawing a teapot from multiple viewpoints.
"""
import argparse
import glob
import os
import subprocess

import chainer
import scipy.misc
import tqdm

import neural_renderer


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--filename_input', type=str, default='./examples/data/teapot.obj')
    parser.add_argument('-o', '--filename_output', type=str, default='./examples/data/example1.gif')
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()
    working_directory = os.path.dirname(args.filename_output)

    # other settings
    camera_distance = 2.732
    elevation = 30

    # load .obj
    # vertices: [num_vertices, XYZ]
    # faces: # [num_faces, 3]
    vertices, faces = neural_renderer.load_obj(args.filename_input)
    vertices = vertices[None, :, :]  #  -> [batch_size=1, num_vertices, XYZ]

    # to gpu
    chainer.cuda.get_device_from_id(args.gpu).use()
    vertices = chainer.cuda.to_gpu(vertices)
    faces = chainer.cuda.to_gpu(faces)

    # create renderer
    renderer = neural_renderer.Renderer()

    # draw object
    loop = tqdm.tqdm(range(0, 360, 4))
    for num, azimuth in enumerate(loop):
        loop.set_description('Drawing')
        renderer.viewpoints = neural_renderer.get_points_from_angles(camera_distance, elevation, azimuth)
        images = renderer.render_silhouettes(vertices, faces)  # [batch_size, RGB, image_size, image_size]
        image = images.data.get()[0]  # [image_size, image_size]
        scipy.misc.toimage(image, cmin=0, cmax=1).save('%s/_tmp_%04d.png' % (working_directory, num))

    # generate gif (need ImageMagick)
    options = '-delay 8 -loop 0 -layers optimize'
    subprocess.call('convert %s %s/_tmp_*.png %s' % (options, working_directory, args.filename_output), shell=True)

    # remove temporary files
    for filename in glob.glob('%s/_tmp_*.png' % working_directory):
        os.remove(filename)


if __name__ == '__main__':
    run()
