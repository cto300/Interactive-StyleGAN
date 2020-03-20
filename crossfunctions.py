import numpy as np
import math
import random
import operators as op

import pickle
import numpy as np
import tensorflow as tf
import PIL.Image

import os
import shutil

import dnnlib
import dnnlib.tflib as tflib
import sys

def init_deepie(url, test, rand = True, seeds = None):
#def init_deepie(generator, test, rand = True, seeds = None):

    tf.InteractiveSession()

    with dnnlib.util.open_url(url) as f:
        _G, _D, Gs = pickle.load(f)

    if (rand == False):
        seeds=[639,701,687,615,2268,444,555,666,789,1092,6745,5653,4543,435,4345,45322,563,785,543,6785]
        latents = np.stack(np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in seeds)

    else:
        #initialice random latents
        rnd = np.random.RandomState(None)
        latents = rnd.randn(20, Gs.input_shape[1])

    # Generate image.
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)

    # Run the generator to produce a set of images.
    l1 = latents[:len(latents)//2]
    l2 = latents[len(latents)//2:]
    images1 = Gs.run(l1, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)
    images2 = Gs.run(l2, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)
    #images1 = generator.run(l1, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)
    #images2 = generator.run(l2, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)
    images = np.concatenate([images1, images2])


    if not os.path.exists("Deepie"):
        os.mkdir("Deepie")

    if not os.path.exists("Deepie/Test%d" % test):
        os.mkdir("Deepie/Test%d" % test)

    if not os.path.exists("Deepie/Test%d/selected" % test):
        os.mkdir("Deepie/Test%d/selected" % test )

    if not os.path.exists("Deepie/Test%d/Generation0" % test):
        os.mkdir("Deepie/Test%d/Generation0" % test )

    for idx in range(images.shape[0]):
        # PIL.Image.fromarray(images[idx], 'RGB').save('Generation0/img%d.png' % idx)
        PIL.Image.fromarray(images[idx], 'RGB').save('Deepie/Test%d/Generation0/img%d.png' % (test,idx))

    return latents

def cross_deepie(url, test, indices, variables, foreign, noise, count):
#def cross_deepie(generator, test, indices, variables, foreign, noise, count):


    tf.InteractiveSession()

    # Generate image.
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)


    with dnnlib.util.open_url(url) as f:
        _G, _D, Gs = pickle.load(f)

    next_pop = op.crossover(variables, foreign, noise, indices)

    l1 = next_pop[:len(next_pop) // 2]
    l2 = next_pop[len(next_pop) // 2:]
    images1 = Gs.run(l1, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)
    images2 = Gs.run(l2, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)
    #images1 = generator.run(l1, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)
    #images2 = generator.run(l2, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)
    images = np.concatenate([images1, images2])

    #save images
    if not os.path.exists("Deepie/Test%d/Generation%d" % (test, count)):
        os.mkdir("Deepie/Test%d/Generation%d" % (test, count))

    for idx in range(images.shape[0]):
        # PIL.Image.fromarray(images[idx], 'RGB').save('Generation0/img%d.png' % idx)
        PIL.Image.fromarray(images[idx], 'RGB').save('Deepie/Test%d/Generation%d/img%d.png' % (test, count, idx))


    return next_pop



def show_images(n, gen, test):

    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid
    import numpy as np
    from PIL import Image

    im_arr = [None] * n

    for idx in range(n):
        im = Image.open(r"Deepie/Test%d/Generation%d/img%d.png" % (test, gen, idx))
        #im_arr[idx] = im
        im_arr[idx] = im.resize((256,256)) 


    fig = plt.figure(figsize=(20, 20))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(4, 5),  # 
                 axes_pad=0.5,  # pad between axes in inch.
                 )

    for idx, (ax, im) in enumerate(zip(grid, im_arr)):
     # Iterating over the grid returns the Axes.
        ax.set_title("img%d.png" % idx)
        ax.imshow(im)

    plt.show()




def deepie(url, test, gen, latents, indices, foreign, noise):


    tf.InteractiveSession()

    with dnnlib.util.open_url(url) as f:
        _G, _D, Gs = pickle.load(f)

     # Generate image.
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)

    if gen == 0:

        #next_pop = None
        #variables = latents.copy()

        # Run the generator to produce a set of images.
        #l1 = variables[:len(variables)//2]
        #l2 = variables[len(variables)//2:]
        l1 = latents[:len(latents)//2]
        l2 = latents[len(latents)//2:]
        images1 = Gs.run(l1, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)
        images2 = Gs.run(l2, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)
        images = np.concatenate([images1, images2])

        if not os.path.exists("Deepie"):
            os.mkdir("Deepie")

        if not os.path.exists("Deepie/Test%d" % test):
            os.mkdir("Deepie/Test%d" % test)

        if not os.path.exists("Deepie/Test%d/selected" % test):
                os.mkdir("Deepie/Test%d/selected" % test )

        if not os.path.exists("Deepie/Test%d/Generation%d" % (test, gen)):
                os.mkdir("Deepie/Test%d/Generation%d" % (test, gen) )

        for idx in range(images.shape[0]):
            # PIL.Image.fromarray(images[idx], 'RGB').save('Generation0/img%d.png' % idx)
            PIL.Image.fromarray(images[idx], 'RGB').save('Deepie/Test%d/Generation%d/img%d.png' % (test, gen, idx))

        show_images(images.shape[0], gen, test)

        gen = gen + 1 

        return latents, gen

    else:

        next_pop = op.crossover(latents, foreign, noise, indices)

        # save selected images
        for idx in indices:
            shutil.copy("Deepie/Test%d/Generation%d/img%d.png" % (test, gen - 1, idx), "Deepie/Test%d/selected" % test)
            old_file = os.path.join("Deepie/Test%d/selected" % (test), "img%d.png" % (idx))
            new_file = os.path.join("Deepie/Test%d/selected" % (test), "img%d_gen%d.png" % (idx, gen - 1))
            os.rename(old_file, new_file)

        l1 = next_pop[:len(next_pop) // 2]
        l2 = next_pop[len(next_pop) // 2:]
        images1 = Gs.run(l1, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)
        images2 = Gs.run(l2, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)
        images = np.concatenate([images1, images2])

        #save images
        if not os.path.exists("Deepie/Test%d/Generation%d" % (test, gen)):
            os.mkdir("Deepie/Test%d/Generation%d" % (test, gen))

        for idx in range(images.shape[0]):
            # PIL.Image.fromarray(images[idx], 'RGB').save('Generation0/img%d.png' % idx)
            PIL.Image.fromarray(images[idx], 'RGB').save('Deepie/Test%d/Generation%d/img%d.png' % (test, gen, idx))

        show_images(images.shape[0], gen, test)

        gen = gen + 1 

        return next_pop, gen


    

def deepie2(generator, test, gen, latents, indices, foreign, noise):


     # Generate image.
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)

    if gen == 0:

        #next_pop = None
        #variables = latents.copy()

        # Run the generator to produce a set of images.
        #l1 = variables[:len(variables)//2]
        #l2 = variables[len(variables)//2:]
        l1 = latents[:len(latents)//2]
        l2 = latents[len(latents)//2:]
        images1 = generator.run(l1, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)
        images2 = generator.run(l2, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)
        images = np.concatenate([images1, images2])

        if not os.path.exists("Deepie"):
            os.mkdir("Deepie")

        if not os.path.exists("Deepie/Test%d" % test):
            os.mkdir("Deepie/Test%d" % test)

        if not os.path.exists("Deepie/Test%d/selected" % test):
                os.mkdir("Deepie/Test%d/selected" % test )

        if not os.path.exists("Deepie/Test%d/Generation%d" % (test, gen)):
                os.mkdir("Deepie/Test%d/Generation%d" % (test, gen) )

        for idx in range(images.shape[0]):
            # PIL.Image.fromarray(images[idx], 'RGB').save('Generation0/img%d.png' % idx)
            PIL.Image.fromarray(images[idx], 'RGB').save('Deepie/Test%d/Generation%d/img%d.png' % (test, gen, idx))

        show_images(images.shape[0], gen, test)

        gen = gen + 1 

        return latents, gen

    else:

        next_pop = op.crossover(latents, foreign, noise, indices)

        # save selected images
        for idx in indices:
            shutil.copy("Deepie/Test%d/Generation%d/img%d.png" % (test, gen - 1, idx), "Deepie/Test%d/selected" % test)
            old_file = os.path.join("Deepie/Test%d/selected" % (test), "img%d.png" % (idx))
            new_file = os.path.join("Deepie/Test%d/selected" % (test), "img%d_gen%d.png" % (idx, gen - 1))
            os.rename(old_file, new_file)

        l1 = next_pop[:len(next_pop) // 2]
        l2 = next_pop[len(next_pop) // 2:]
        images1 = generator.run(l1, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)
        images2 = generator.run(l2, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)
        images = np.concatenate([images1, images2])

        #save images
        if not os.path.exists("Deepie/Test%d/Generation%d" % (test, gen)):
            os.mkdir("Deepie/Test%d/Generation%d" % (test, gen))

        for idx in range(images.shape[0]):
            # PIL.Image.fromarray(images[idx], 'RGB').save('Generation0/img%d.png' % idx)
            PIL.Image.fromarray(images[idx], 'RGB').save('Deepie/Test%d/Generation%d/img%d.png' % (test, gen, idx))

        show_images(images.shape[0], gen, test)

        gen = gen + 1 

        return next_pop, gen

  


