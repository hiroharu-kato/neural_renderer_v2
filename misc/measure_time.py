from __future__ import print_function
import argparse
import time

import chainer
import numpy as np
import tqdm

import neural_renderer


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--filename_input', type=str, default='./examples/data/teapot.obj')
    parser.add_argument('-bs', '--batch_size', type=int, default=1)
    parser.add_argument('-is', '--image_size', type=int, default=256)
    parser.add_argument('-us', '--unsafe', type=int, default=0)
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()

    # other settings
    camera_distance = 2.732
    elevation = 30
    texture_size = 2

    # load .obj
    vertices, faces = neural_renderer.load_obj(args.filename_input)
    vertices = vertices[None, :, :]  # [num_vertices, XYZ] -> [batch_size=1, num_vertices, XYZ]

    # create texture [batch_size=1, num_faces, texture_size, texture_size, texture_size, RGB]
    vertices_t, faces_t, textures = neural_renderer.create_textures(faces.shape[0])

    # tile to minibatch
    vertices = np.tile(vertices, (args.batch_size, 1, 1))
    vertices_t = np.tile(vertices_t, (args.batch_size, 1, 1))
    textures = np.tile(textures, (args.batch_size, 1, 1, 1))

    # to gpu
    chainer.cuda.get_device_from_id(args.gpu).use()
    vertices = chainer.Variable(chainer.cuda.to_gpu(vertices))
    faces = chainer.cuda.to_gpu(faces)
    vertices_t = chainer.Variable(chainer.cuda.to_gpu(vertices_t))
    faces_t = chainer.cuda.to_gpu(faces_t)
    textures = chainer.Variable(chainer.cuda.to_gpu(textures))

    # create renderer
    renderer = neural_renderer.Renderer()
    renderer.image_size = args.image_size

    # draw object
    times_forward = []
    times_backward = []
    loop = tqdm.tqdm(range(0, 360, 15))
    for num, azimuth in enumerate(loop):
        loop.set_description('Drawing')
        renderer.viewpoints = neural_renderer.get_points_from_angles(camera_distance, elevation, azimuth)
        time_start = time.time()
        images = renderer.render_silhouettes(vertices, faces)  # [batch_size, image_size, image_size]
        _ = images.data[0, 0, 0].get()
        time_end = time.time()
        times_forward.append(time_end - time_start)
        loss = chainer.functions.sum(images)
        _ = loss.data.get()
        time_start = time.time()
        loss.backward()
        time_end = time.time()
        times_backward.append(time_end - time_start)

    print('silhouette forward time: %.3f ms' % (np.sum(times_forward[1:]) / len(times_forward[1:])))
    print('silhouette backward time: %.3f ms' % (np.sum(times_backward[1:]) / len(times_backward[1:])))

    # draw object
    times_forward = []
    times_backward = []
    loop = tqdm.tqdm(range(0, 360, 15))
    for num, azimuth in enumerate(loop):
        loop.set_description('Drawing')
        renderer.viewpoints = neural_renderer.get_points_from_angles(camera_distance, elevation, azimuth)
        time_start = time.time()
        images = renderer.render(vertices, faces, vertices_t, faces_t, textures)  # [batch_size, RGB, image_size, image_size]
        _ = images.data[0, 0, 0, 0].get()
        time_end = time.time()
        times_forward.append(time_end - time_start)
        loss = chainer.functions.sum(images)
        _ = loss.data.get()
        time_start = time.time()
        loss.backward()
        time_end = time.time()
        times_backward.append(time_end - time_start)

    print('texture forward time: %.3f ms' % (np.sum(times_forward[1:]) / len(times_forward[1:])))
    print('texture backward time: %.3f ms' % (np.sum(times_backward[1:]) / len(times_backward[1:])))


if __name__ == '__main__':
    from chainer.function_hooks import TimerHook
    hook = TimerHook()
    import cupy
    from cupy import prof

    with cupy.cuda.profile():
        with cupy.prof.time_range('some range in green', color_id=0):
            with hook:
                run()
            hook.print_report()
            print(hook.total_time())