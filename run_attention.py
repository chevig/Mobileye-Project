try:
    print("Elementary imports: ")
    import os
    import json
    import glob
    import argparse

    print("numpy/scipy imports:")
    import numpy as np
    from scipy import signal as sg
    import scipy.ndimage as ndimage
    from scipy.ndimage.filters import maximum_filter

    print("PIL imports:")
    from PIL import Image

    print("matplotlib imports:")
    import matplotlib.pyplot as plt
except ImportError:
    print("Need to fix the installation")
    raise

print("All imports okay. Yay!")


def red_convolution(img):
    kernel = np.ones((3, 3)) / -9
    kernel[1, 1] = 8 / 9
    return ndimage.convolve(img[:, :, 0], kernel)


def green_convolution(img):
    kernel = np.ones((3, 3)) / -9
    kernel[1, 1] = 8 / 9
    return ndimage.convolve(img[:, :, 1], kernel)


def find_tfl_lights(c_image: np.ndarray, **kwargs):
    """
    Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement
    :param c_image: The image itself as np.uint8, shape of (H, W, 3)
    :param kwargs: Whatever config you want to pass in here
    :return: 4-tuple of x_red, y_red, x_green, y_green
    """
    # my code here
    image = np.array(Image.open(c_image))
    c_image = image.copy() / 255

    convolved_red = red_convolution(c_image)
    convolved_green = green_convolution(c_image)

    red_filtered_img = ndimage.maximum_filter(convolved_red, 50)
    green_filtered_img = ndimage.maximum_filter(convolved_green, 50)

    red_position = np.argwhere(red_filtered_img == convolved_red)
    red_position = list(filter(lambda l: c_image[l[0], l[1], 0] > 0.8, red_position))

    green_position = np.argwhere(green_filtered_img == convolved_green)
    green_position = list(filter(lambda l: c_image[l[0], l[1], 1] > 0.8, green_position))

    x_red = [p[0] for p in red_position]
    y_red = [p[1] for p in red_position]

    x_green = [p[0] for p in green_position]
    y_green = [p[1] for p in green_position]

    # return x_red, y_red, x_green, y_green
    return y_red, x_red, y_green, x_green
    # return [],[],[],[]


def show_image_and_gt(image, objs, fig_num=None):
    plt.figure(fig_num).clf()
    plt.imshow(image)
    labels = set()
    if objs is not None:
        for o in objs:
            poly = np.array(o['polygon'])[list(np.arange(len(o['polygon']))) + [0]]
            plt.plot(poly[:, 0], poly[:, 1], 'r', label=o['label'])
            labels.add(o['label'])
        if len(labels) > 1:
            plt.legend()


def test_find_tfl_lights(image_path, json_path=None, fig_num=None):
    """
    Run the attention code
    """
    image = np.array(Image.open(image_path))
    if json_path is None:
        objects = None
    else:
        gt_data = json.load(open(json_path))
        what = ['traffic light']
        objects = [o for o in gt_data['objects'] if o['label'] in what]

    show_image_and_gt(image, objects, fig_num)

    red_x, red_y, green_x, green_y = find_tfl_lights(image, some_threshold=42)
    plt.plot(red_x, red_y, '+', color='r', markersize=4)
    plt.plot(green_x, green_y, '+', color='g', markersize=4)
    # plt.plot(red_x, red_y, 'ro', color='r', markersize=4)
    # plt.plot(green_x, green_y, 'ro', color='g', markersize=4)


def main(argv=None):
    """It's nice to have a standalone tester for the algorithm.
    Consider looping over some images from here, so you can manually exmine the results
    Keep this functionality even after you have all system running, because you sometime want to debug/improve a module
    :param argv: In case you want to programmatically run this"""
    parser = argparse.ArgumentParser("Test TFL attention mechanism")
    parser.add_argument('-i', '--image', type=str, help='Path to an image')
    parser.add_argument("-j", "--json", type=str, help="Path to json GT for comparison")
    parser.add_argument('-d', '--dir', type=str, help='Directory to scan images in')
    args = parser.parse_args(argv)
    default_base = '../../data'

    if args.dir is None:
        args.dir = default_base
    flist = glob.glob(os.path.join(args.dir, '*_leftImg8bit.png'))

    for image in flist:
        json_fn = image.replace('_leftImg8bit.png', '_gtFine_polygons.json')

        if not os.path.exists(json_fn):
            json_fn = None
        test_find_tfl_lights(image, json_fn)

    if len(flist):
        print("You should now see some images, with the ground truth marked on them. Close all to quit.")

    else:
        print("Bad configuration?? Didn't find any picture to show")

    plt.show(block=True)
#
#
# if __name__ == '__main__':
#     main()