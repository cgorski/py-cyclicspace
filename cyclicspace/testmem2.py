import numba
rtsys = numba.runtime.rtsys
import psutil
import numpy as np  



@numba.njit
def jit_calc_cycle(o_uni, nv, h, w):
    n_uni = np.zeros((h, w))
    count = 1
    for y in range(0, h):
        for x in range(0, w):
            n = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            neighbors = [((p[0] + y) % h, (p[1] + x) % w) for p in n]
            for e in neighbors:
                found = False
                y2 = e[0]
                x2 = e[1]

                if o_uni[y][x] == (o_uni[y2][x2] + 1) % nv:
                    found = True
                    n_uni[y][x] = o_uni[y2][x2]
                    break
                if not found:
                    n_uni[y][x] = o_uni[y][x]
                count +=1
    return n_uni

def main():
    num_values = 8
    height = 100
    width = 100
    
    num_array = np.random.randint(num_values, size=(height, width))

    for i in range(10):
        num_array = jit_calc_cycle(num_array, num_values, height, width)
        print(proc.memory_full_info())
        print(rtsys.get_allocation_stats())

if __name__ == "__main__":
    proc = psutil.Process()
    print(proc.memory_full_info())
    main()

