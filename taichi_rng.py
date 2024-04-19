''''
Using np.random in taichi since taichi's random number generator is bad
'''
import taichi as ti
import numpy as np

if __name__ == "__main__":
    ti.init(arch=ti.cpu, default_fp=ti.f64)

NUM_RANDOM_NUMBERS = 100000
# This type has scope in taichi and python
INDEX = ti.field(dtype=ti.i32, shape=())
INDEX[None] = 0
RANDOM_NUMBERS = ti.field(dtype=ti.f64, shape=NUM_RANDOM_NUMBERS)

@ti.func
def uniform_float() -> ti.f64:
    INDEX[None] = (INDEX[None] + 1) % NUM_RANDOM_NUMBERS
    return RANDOM_NUMBERS[INDEX[None] - 1]

@ti.func
def randint(a: ti.i32, b: ti.i32) -> ti.i32:
    return int(a + uniform_float() * (b - a))

# @ti.func
# def sample(population, k: ti.i32) -> ti.Vector.field(2, dtype=ti.f64):
#     result = ti.Vector.field(2, dtype=ti.f64, shape=(k,))
#     for i in range(k):
#         random_int = randint(0, len(population)-1)
#         result[i] = population[random_int]
#     return result

def generate_random():
    file = open("random.txt", "w")
    for i in range(NUM_RANDOM_NUMBERS):
        file.write(str(np.random.random()) + "\n")
    file.close()
    
def read_random():
    with open("random.txt", "r") as file:
        numbers = list(map(float, file.readlines()))
        indices = np.arange(NUM_RANDOM_NUMBERS)
        np.random.shuffle(indices)
        for i in range(NUM_RANDOM_NUMBERS):
            RANDOM_NUMBERS[i] = numbers[indices[i]]

@ti.kernel
def test_uniform_float():
    print(uniform_float())
    
read_random()
if __name__ == '__main__':
    generate_random()
    test_uniform_float()
    