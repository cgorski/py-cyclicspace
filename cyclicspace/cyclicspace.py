from numba import njit, cuda  # type: ignore
from numba.typed import List  # type: ignore

import argparse
import typing
import gc

import threading

import colorsys
import math
import numpy as np  # type: ignore
import pprint as pp


from PIL import Image  # type: ignore


@njit
def colorval_to_rgb(color_val: int, max_val: int) -> typing.Tuple[int, int, int]:
    # assert color_val >= 0, "color_val must be non-negative, given value is {0}".format(
    #     color_val
    # )
    # assert max_val >= 0, "max_val must be non-negative, given value is {0}".format(
    #     max_val
    # )
    # assert (
    #     max_val >= color_val
    # ), "max_val must be greater than or equal to color_val, max_val is {0} and color_val is {1}".format(
    #     max_val, color_val
    # )
    r, g, b = colorsys.hsv_to_rgb(float(color_val) / float(max_val), 1, 1)
    return (math.floor(r * 255), math.floor(g * 255), math.floor(b * 255))


def seed_generator(seed: int = None) -> int:
    if seed:
        s = seed
        np.random.seed(seed=s)
    else:
        np.random.seed()
        s = np.random.randint((2 ** 32), size=1)[0]
        np.random.seed(seed=s)
    return s

@njit
def hsv_to_rgb(max_val: int, color_val: int, s: float = 1.0, v: float = 1.0):
    """color_val is h converted using max_val, s/v are [0,1]"""
    h = (color_val/max_val) * 360.0
    c = v * s
    x = c * (1.0 - abs(((h / 60.0) % 2) - 1.0))
    m = v - c

    if (h >= 0.0 and h < 60.0):
        r,g,b = (c + m, x + m, m)
    elif (h >= 60.0 and h < 120.0):
        r,g,b = (x + m, c + m, m)
    elif (h >= 120.0 and h < 180.0):
        r,g,b = (m, c + m, x + m)
    elif (h >= 180.0 and h < 240.0):
        r,g,b = (m, x + m, c + m)
    elif (h >= 240.0 and h < 300.0):
        r,g,b = (x + m, m, c + m)
    elif (h >= 300.0 and h < 360.0):
        r,g,b = (c + m, m, x + m)
    else:
        r,g,b = (m, m, m)

    return [math.floor(r*255), math.floor(g*255), math.floor(b*255)]
    

@njit
def color_array(max_val, arr):
    shape = arr.shape
    h = shape[0]
    w = shape[1]
    colors = np.zeros((h,w,3), dtype=np.uint8)
    for y in range(0,h):
        for x in range(0,w):
            colors[y][x] = hsv_to_rgb(max_val, arr[y][x])
    return colors


def random_array(num_values, width, height):
    init_universe = np.random.randint(num_values, size=(height, width))
    return init_universe




def colorbar_image():
    h, w = 1024, 1024
    imagedata = np.zeros((h, w, 3), dtype=np.uint8)
    for line in range(0, h):
        imagedata[line, 0:w] = hsv_to_rgb(h - 1, line)
    return imagedata



@njit
def jit_calc_cycle(o_uni, n_uni, nv, w, h):
    for y in range(0, h):
        for x in range(0, w):
            n = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            neighbors = [((p[0] + y) % h, (p[1] + x) % w) for p in n]
            for e in neighbors:
                y2 = e[0]
                x2 = e[1]
                if o_uni[y][x] == (o_uni[y2][x2] + 1) % nv:
                    n_uni[y][x] = o_uni[y2][x2]
                    break
                else:
                    n_uni[y][x] = o_uni[y][x]
    return




class Universe:
    num_values: int

    def __init__(
        self, num_values: int, width: int, height: int, random: bool = True
    ) -> None:
        if random:
            self.universe_array = random_array(num_values, width, height)
        else:
            self.universe_array = np.zeros((height, width))

        self.num_values = num_values

    def height(self) -> int:
        return self.universe_array.shape[0]

    def width(self) -> int:
        return self.universe_array.shape[1]

    def next_cycle(self) -> "Universe":
        new_universe = Universe(
            num_values=self.num_values,
            width=self.width(),
            height=self.height(),
            random=False,
        )

        n_uni = new_universe.universe_array
        o_uni = self.universe_array
        nv = self.num_values
        w = self.width()
        h = self.height()

        jit_calc_cycle(o_uni=o_uni, n_uni=n_uni, nv=nv, w=w, h=h)
        
        return new_universe


    def image(self: "Universe"):
        nv = self.num_values
        arr = self.universe_array
        image_data = color_array(nv, arr)
#        image_data = colorbar_image()
        return Image.fromarray(image_data, "RGB")

    def neighbors(self: 'Universe', y, x, w, h):
        n = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for p in n:
            yield ((p[0] + y) % h, (p[1] + x) % w)


    def iter(self: "Universe") -> "typing.Iterator[Universe]":
        next = self
        while True:
            yield next
            next = next.next_cycle()
    



def parse_args():
    parser = argparse.ArgumentParser(description='Compute iterations of cyclic space.')
    parser.add_argument("filename_prefix", metavar="FILENAME_PREFIX", help='The filename prefix for generated PNG images.')
    parser.add_argument("cell_values", metavar="CELL_VALUES", help="The number of valid cell values, from 0 to CELL_VALUES-1.", type=int)
    parser.add_argument("max_images", metavar="MAX_IMAGES", help="The maximum number of images to generate.", type=int)
    parser.add_argument("width", metavar='WIDTH', help='The width of the cyclic space universe.', type=int)
    parser.add_argument("height", metavar='HEIGHT', help='The height of the cyclic space universe.', type=int)

    parser.add_argument("-s", "--step_size", help='The number of generations between image files, default is 1.', default=1, type=int)
    args = parser.parse_args()
    print(args.filename_prefix)
    return args

def main():
    args = parse_args()
    
    pp.pprint(seed_generator())
    print("Creating initial universe...")
    uni = Universe(num_values=args.cell_values, width=args.width, height=args.height, random=True)
    file_num = 0
    step = 0
    for i in uni.iter():

        print('...done.')
        if (step % args.step_size) == 0:

            filename = "{}-{:08d}-{:08d}.png".format(args.filename_prefix, file_num, step)
            print("Saving file: {}".format(filename))
            img = i.image()
            img.save(filename)
            print("...done.")
            if file_num == (args.max_images-1):
                print("Reached max images. Complete.")
                break
            file_num += 1
        step+=1
        print("Starting calculation...")

            


if __name__ == "__main__":
    main()
#    colorbar_image()
