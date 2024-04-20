import taichi as ti
if __name__ == "__main__":
	ti.init(ti.cpu, default_fp=ti.f64)
 
from taichi_rng import randint

'''Fetching Data'''
def read_and_convert_to_dict(FILE_PATH):
	data_dict = {}
	city_list = []
	with open(FILE_PATH, 'r') as file:
		for line in file:
			# Split the line into parts
			parts = line.strip().split()
			try:
				# Extract key and coordinates
				key = int(parts[0])
				# adding the city to the list as well to keep record
				city_list.append(key)
				coordinates = tuple(map(float, parts[1:]))
				# Create dictionary entry
				data_dict[key] = coordinates
			except:
				continue
	return city_list, data_dict

FILE_PATH = 'data/qa194.tsp'  # Replace with the path to your text file
city_keys, city_dict = read_and_convert_to_dict(FILE_PATH)

# need to initialize taichi before importing this file
if __name__ == "__main__":
	ti.init(arch=ti.cpu, default_fp=ti.f64)

NUM_CITIES = len(city_keys)

# # BUG: Taichi gives a warning to use vectors larger than size 32
TYPE_GENOME = ti.types.vector(NUM_CITIES, ti.i32)

CITY_KEYS = ti.field(dtype=ti.i32, shape=len(city_keys))
CITY_COORDS = ti.Vector.field(2, dtype=ti.f64, shape=len(city_dict))

for i in range(len(city_keys)):
    CITY_KEYS[i] = city_keys[i]
    CITY_COORDS[i][0], CITY_COORDS[i][1] = city_dict[city_keys[i]]

@ti.func
def get_distance(x: ti.math.vec2, y: ti.math.vec2) -> ti.f64:
    # x and y are two 2D points each not 2 coordinates of one 2D point.
    return ((x[0]-y[0])**2 + (x[1]-y[1])**2)**(1/2)

@ti.func
def distance(cities, num_cities) -> ti.f64:
	distance = ti.f64(0.0)
	for i in range(num_cities):
		a, b = cities[i], cities[(i+1) % num_cities]
		distance += get_distance(CITY_COORDS[a-1], CITY_COORDS[b-1])
	return distance

@ti.func
def _generate_genome() -> TYPE_GENOME:
	genome = TYPE_GENOME([i+1 for i in range(NUM_CITIES)])
	for i in range(NUM_CITIES):
		j = randint(0, NUM_CITIES-1)
		genome[i], genome[j] = genome[j], genome[i]
	return genome

@ti.kernel
def test_kernel():
	
	print(distance(CITY_KEYS, NUM_CITIES))

if __name__ == "__main__": 
    print(CITY_KEYS)   
    test_kernel()