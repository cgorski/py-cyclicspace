from numba import jit, prange

import typing


import colorsys
import math
import numpy as np  # type: ignore
import pprint as pp


from PIL import Image  # type: ignore


def colorval_to_rgb(color_val: int, max_val: int) -> typing.Tuple[int, int, int]:
    assert color_val >= 0, "color_val must be non-negative, given value is {0}".format(
        color_val
    )
    assert max_val >= 0, "max_val must be non-negative, given value is {0}".format(
        max_val
    )
    assert (
        max_val >= color_val
    ), "max_val must be greater than or equal to color_val, max_val is {0} and color_val is {1}".format(
        max_val, color_val
    )
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


def random_array(num_values, width, height):
    init_universe = np.random.randint(num_values, size=(height, width))
    return init_universe


def colorbar_image():
    h, w = 1024, 1024
    imagedata = np.zeros((h, w, 3), dtype=np.uint8)
    for line in range(0, h):
        imagedata[line, 0:w] = list(colorval_to_rgb(line, h - 1))
    img = Image.fromarray(imagedata, "RGB")
    img.save("colorbar.png")



@jit(nopython=True)#, parallel=True)    
def neighbors(y,x,w,h):
#    n = [(-1,-1),(-1,0),(-1,1),
#         (0,-1),(0,1),
#         (1,-1),(1,0), (1,1)]
    n = [(-1,0),(1,0),(0,-1),(0,1)]
    for p in n:
        yield ((p[0]+y)%h,(p[1]+x)%w)
    
@jit(nopython=True)#, parallel=True)
def calc_cycle(o_uni, n_uni, nv, w, h):
    for y in prange(0,h):
        for x in prange(0,w):
            for e in neighbors(y=y,x=x,w=w,h=h):
                found = False
                y2 = e[0]
                x2 = e[1]
                if(o_uni[y][x] == (o_uni[y2][x2]+1) % nv):
                    found = True
                    n_uni[y][x] = o_uni[y2][x2]
                    break
            if not found:
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
            self.universe_array = np.zeros((height,width))

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

        calc_cycle(o_uni = o_uni, n_uni = n_uni, nv = nv, w = w, h = h)

        return new_universe

    
    def iter(self: 'Universe') -> 'typing.Iterator[Universe]':
        next = self
        while True:
            yield next
            next = next.next_cycle()
            
    def image(self: 'Universe'):
        image_data = np.zeros((self.height(), self.width(), 3), dtype=np.uint8)
        it = np.nditer(self.universe_array, flags=["multi_index"])
        while not it.finished:
            y = it.multi_index[0]
            x = it.multi_index[1]
            image_data[y][x]=colorval_to_rgb(self.universe_array[y][x], self.num_values)
            it.iternext()

        return Image.fromarray(image_data, "RGB")


import cProfile
    
def main():
    pp.pprint(seed_generator())
    uni = Universe(num_values=16, width=300, height=300, random=True)
    b = 0
    for i in uni.iter():
        b = b+1
        print ("started " + "image-{:08d}.png".format(b))
        i.image().save("img/image-{:08d}.png".format(b))
        print ("saved " + "image-{:08d}.png".format(b))
        print()





    
if __name__ == "__main__":
    main()
#    colorbar_image()
