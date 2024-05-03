''''
Using np.random in taichi since taichi's random number generator is bad
'''
import taichi as ti
import numpy as np

if __name__ == "__main__":
    ti.init(arch=ti.cpu, default_fp=ti.f64)

NUM_RANDOM_NUMBERS = 100000
NUM_SEQUENCES = 10
# This type has scope in taichi and python
INDEX = ti.field(dtype=ti.i32, shape=())
INDEX[None] = 0
RANDOM_NUMBERS = ti.field(dtype=ti.f64, shape=(NUM_SEQUENCES, NUM_RANDOM_NUMBERS))

@ti.func
def uniform_float() -> ti.f64:
    INDEX[None] = (INDEX[None] + 1) % NUM_RANDOM_NUMBERS
    return RANDOM_NUMBERS[0, INDEX[None] - 1]

@ti.func
def uniform_float_isl(seq: ti.i32) -> ti.f64:
    INDEX[None] = (INDEX[None] + 1) % NUM_RANDOM_NUMBERS
    return RANDOM_NUMBERS[seq%NUM_SEQUENCES, INDEX[None] - 1]

@ti.func
def randint(a: ti.i32, b: ti.i32) -> ti.i32:
    return int(a + uniform_float() * (b - a))

@ti.func
def randint_isl(a: ti.i32, b: ti.i32, seq: ti.i32) -> ti.i32:
    return int(a + uniform_float_isl(seq) * (b - a))

@ti.func
def randfloat_isl(a: ti.f64, b: ti.f64, seq: ti.i32) -> ti.f64:
    return float(a + uniform_float_isl(seq) * (b - a))



def generate_random():
    file = open("random.txt", "w")
    for i in range(NUM_RANDOM_NUMBERS*NUM_SEQUENCES):
        file.write(str(np.random.random()) + "\n")
    file.close()
    
def read_random():
    with open("random.txt", "r") as file:
        numbers = list(map(float, file.readlines()))
        for seq in range(NUM_SEQUENCES):
            # print(len(numbers))
            indices = np.arange(NUM_RANDOM_NUMBERS)
            np.random.shuffle(indices)
            for i in range(NUM_RANDOM_NUMBERS):
                RANDOM_NUMBERS[seq, i] = numbers[seq*NUM_RANDOM_NUMBERS + indices[i]]

@ti.kernel
def test_uniform_float():
    print(uniform_float())
    print(uniform_float_isl(2))

read_random()
if __name__ == '__main__':
    # generate_random()
    read_random()
    test_uniform_float()
    print(RANDOM_NUMBERS)
    
    