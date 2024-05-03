import taichi as ti
if __name__ == "__main__":
	ti.init(arch=ti.cpu, default_fp=ti.f64)
from taichi_rng import *
	
from taichi_tsp_readdata import NUM_CITIES, distance, TYPE_GENOME, _generate_genome, _generate_genome_isl

LCS_Buffer = ti.field(dtype=ti.i32, shape=(NUM_CITIES + 1, NUM_CITIES + 1))

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
	def initialize_isl(self, isl_ind):
		self.genome_size = NUM_CITIES
		self.genome = _generate_genome_isl(isl_ind)
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
	
	@ti.func
	def mutate_isl(self, isl_ind) -> None:
		rand_index1 = randint_isl(0, self.genome_size, isl_ind)
		rand_index2 = randint_isl(0, self.genome_size, isl_ind)
		self.genome[rand_index1], self.genome[rand_index2] = self.genome[rand_index2], self.genome[rand_index1]
		distance = distance(self.genome, self.genome_size)
		if rand_index1 != rand_index2:
			if ti.abs(distance - self.fitness) < 0.001:
				print("ERROR: mutation not working properly")
		self.fitness = distance
		
	@ti.func
	def euclidean_distance(self, individual):
		return ti.math.distance(self.genome, individual.genome)

	@ti.func
	def hamming_distance(self, individual):
		difference = 0
		for i in range(NUM_CITIES):
			if(self.genome[i] != individual.genome[i]):
				difference += 1
		return difference
	
	@ti.func
	def LCS(self, individual):		
		# declaring the array for storing the dp values 		
		"""Following steps build L[m + 1][n + 1] in bottom up fashion 
		Note: L[i][j] contains length of LCS of X[0..i-1] 
		and Y[0..j-1]"""
		for i in range(NUM_CITIES + 1): 
			for j in range(NUM_CITIES + 1): 
				if i == 0 or j == 0 : 
					LCS_Buffer[i, j] = 0
				elif self.genome[i-1] == individual.genome[j-1]: 
					LCS_Buffer[i, j] = LCS_Buffer[i-1, j-1]+1
				else: 
					LCS_Buffer[i, j] = ti.math.max(LCS_Buffer[i-1, j], LCS_Buffer[i, j-1]) 
	
		# L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1] 
		return LCS_Buffer[NUM_CITIES, NUM_CITIES]	


@ti.func
def TSP_random_length_crossover(self, parent1, parent2, isl_ind:int):
	start = randint_isl(0, NUM_CITIES-2, isl_ind)
	end = randint_isl(start, NUM_CITIES-1, isl_ind)

	offspring1_genome = ti.Vector([-1 for _ in range(NUM_CITIES)], dt=ti.i32)
	offspring2_genome = ti.Vector([-1 for _ in range(NUM_CITIES)], dt=ti.i32)

	for i in range(start, end+1):
		offspring1_genome[i] = parent1.genome[i]
		offspring2_genome[i] = parent2.genome[i]

	parent1_pointer = (end + 1) % NUM_CITIES
	parent2_pointer = (end + 1) % NUM_CITIES
	# There's a more efficient way to do this I'm sure
	pointer = (end + 1) % NUM_CITIES
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

	pointer = (end + 1) % NUM_CITIES
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
	