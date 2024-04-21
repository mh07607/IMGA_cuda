import taichi as ti
if __name__ == "__main__":
	ti.init(arch=ti.cpu, default_fp=ti.f64)
from taichi_rng import *
	
from taichi_tsp_readdata import NUM_CITIES, distance, TYPE_GENOME, _generate_genome




@ti.dataclass
class Individual:
	genome: TYPE_GENOME
	fitness: ti.f64
	genome_size: ti.i32
	
	@ti.func 
	def initialize(self):
		self.genome_size = NUM_CITIES
		self.genome = _generate_genome()
		self.fitness = distance(self.genome, self.genome_size)
		
	@ti.func
	def initialize_with_genome(self, genome):
		self.genome_size = NUM_CITIES
		self.genome = genome
		self.fitness = distance(self.genome, self.genome_size)
		
	@ti.func
	def mutate(self) -> None:
		rand_index1 = randint(0, self.genome_size)
		rand_index2 = randint(0, self.genome_size)
		self.genome[rand_index1], self.genome[rand_index2] = self.genome[rand_index2], self.genome[rand_index1]
		distance = distance(self.genome, self.genome_size)
		if rand_index1 != rand_index2:
			if ti.abs(distance - self.fitness) < 0.001:
				print("ERROR: mutation not working properly")
		self.fitness = distance
		
@ti.dataclass
class Individual_2_tuple:
	first: Individual
	second: Individual
	
# def TSP_random_length_crossover(parent1: TSP_Path, parent2: TSP_Path):
# 	length = len(parent1.genome)
# 	start = random.randint(1, length-3)
# 	end = random.randint(start, length-2)

# 	offspring1 = [None] * length
# 	offspring2 = [None] * length

# 	offspring1[start:end+1] = parent1.genome[start:end+1]
# 	offspring2[start:end+1] = parent2.genome[start:end+1]

# 	pointer = end + 1
# 	parent1_pointer = end + 1
# 	parent2_pointer = end + 1

# 	while None in offspring1:
# 		if parent2.genome[parent2_pointer] not in offspring1:
# 			offspring1[pointer % length] = parent2.genome[parent2_pointer]
# 			pointer += 1
# 		parent2_pointer = (parent2_pointer + 1) % length

# 	pointer = 0

# 	while None in offspring2:
# 		if parent1.genome[parent1_pointer] not in offspring2:
# 			offspring2[pointer % length] = parent1.genome[parent1_pointer]
# 			pointer += 1
# 		parent1_pointer = (parent1_pointer + 1) % length

# 	offspring1 = TSP_Path(offspring1)
# 	offspring2 = TSP_Path(offspring2)

# 	return offspring1, offspring2

@ti.func 
def IN(iterable, length, element) -> ti.i32:
	ret = 0
	for i in range(length):
		if iterable[i] == element:
			ret = 1
	return ret

@ti.func
def TSP_random_length_crossover(self, parent1, parent2):
	start = randint(1, NUM_CITIES-3)
	end = randint(start, NUM_CITIES-2)

	offspring1_genome = ti.Vector([-1 for _ in range(NUM_CITIES)], dt=ti.i32)
	offspring2_genome = ti.Vector([-1 for _ in range(NUM_CITIES)], dt=ti.i32)

	for i in range(start, end+1):
		offspring1_genome[i] = parent1.genome[i]
		offspring2_genome[i] = parent2.genome[i]

	parent1_pointer = end + 1
	parent2_pointer = end + 1
	# There's a more efficient way to do this I'm sure
	pointer = end + 1
	count = end-start+1
	while count < NUM_CITIES:
		gene_found = 0
		for i in range(NUM_CITIES):
			if(offspring1_genome[i] == parent2.genome[parent2_pointer]):
				gene_found = 1
				break

		if not gene_found:
			offspring1_genome[pointer % NUM_CITIES] = parent2.genome[parent2_pointer]
			count += 1
			pointer += 1
		parent2_pointer = (parent2_pointer + 1) % NUM_CITIES

	pointer = end+1
	count = end-start+1
	while count < NUM_CITIES:				
		gene_found = 0
		for i in range(NUM_CITIES):
			if(offspring2_genome[i] == parent1.genome[parent1_pointer]):				
				gene_found = 1
				break

		if not gene_found:			
			offspring2_genome[pointer % NUM_CITIES] = parent1.genome[parent1_pointer]
			count += 1
			pointer += 1
		parent1_pointer = (parent1_pointer + 1) % NUM_CITIES  
	
	return offspring1_genome, offspring2_genome

@ti.kernel
def test_kernel():
	individual = Individual()
	individual.genome[23] = 90
	print(individual.genome)
	individual.initialize()
	# # BUG: Uncommenting this line will make the individual.fitness to not change in mutate ??? (Weird)
	# # print(individual.genome)
	print("fitness: ", individual.fitness)
	individual.mutate()
	individual.genome[193] = 900
	# print("genome: ", individual.genome.val)
	print("fitness: ", individual.fitness)
	print(individual.genome)

@ti.kernel
def test_crossover():
	individual1 = Individual()
	individual1_genome = ti.Vector([(i+1) for i in range(NUM_CITIES)])
	individual1.initialize_with_genome(individual1_genome)

	individual2 = Individual()
	individual2_genome = ti.Vector([(NUM_CITIES) - i for i in range(NUM_CITIES)])
	individual2.initialize_with_genome(individual2_genome)

	print(individual1.genome)
	print(individual2.genome)
	print("performing crossover")
	print(TSP_random_length_crossover(individual1, individual2))

if __name__ == "__main__":
	test_crossover()
	