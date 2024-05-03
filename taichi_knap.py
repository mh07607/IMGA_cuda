import taichi as ti
if __name__ == "__main__":
	ti.init(arch=ti.cpu, default_fp=ti.f64)
from taichi_rng import *
	
from taichi_knap_readdata import TOTAL_ITEMS, TYPE_GENOME, calculate_fitness, _generate_genome_isl

LCS_Buffer = ti.field(dtype=ti.i32, shape=(TOTAL_ITEMS + 1, TOTAL_ITEMS + 1))

@ti.dataclass
class Individual:
	genome: TYPE_GENOME
	fitness: ti.f64
	genome_size: ti.i32
	
	# @ti.func 
	# def initialize(self):
	# 	self.genome_size = TOTAL_ITEMS
	# 	self.genome = _generate_genome()
	# 	self.fitness = fitness(self.genome, self.genome_size)
  
	@ti.func
	def initialize_isl(self, isl_ind):
		self.genome_size = TOTAL_ITEMS
		self.genome = _generate_genome_isl(isl_ind)
		self.fitness = calculate_fitness(self.genome)
		
	@ti.func
	def initialize_with_genome(self, genome):
		self.genome_size = TOTAL_ITEMS
		self.genome = genome
		self.fitness = calculate_fitness(self.genome)
		
	@ti.func
	def mutate(self) -> None:
		rand_index1 = randint(0, self.genome_size)		
		self.genome[rand_index1] = 1 - self.genome[rand_index1]
		fitness = calculate_fitness(self.genome)
		self.fitness = fitness
	
	@ti.func
	def mutate_isl(self, isl_ind) -> None:
		rand_index1 = randint_isl(0, self.genome_size, isl_ind)		
		self.genome[rand_index1] = 1 - self.genome[rand_index1]
		fitness = calculate_fitness(self.genome)
		self.fitness = fitness
		
	@ti.func
	def euclidean_distance(self, individual):
		return ti.math.distance(self.genome, individual.genome)

	@ti.func
	def hamming_distance(self, individual):
		difference = 0
		for i in range(TOTAL_ITEMS):
			if(self.genome[i] != individual.genome[i]):
				difference += 1
		return difference
	
	@ti.func
	def LCS(self, individual):		
		# declaring the array for storing the dp values 		
		"""Following steps build L[m + 1][n + 1] in bottom up fashion 
		Note: L[i][j] contains length of LCS of X[0..i-1] 
		and Y[0..j-1]"""
		for i in range(TOTAL_ITEMS + 1): 
			for j in range(TOTAL_ITEMS + 1): 
				if i == 0 or j == 0 : 
					LCS_Buffer[i, j] = 0
				elif self.genome[i-1] == individual.genome[j-1]: 
					LCS_Buffer[i, j] = LCS_Buffer[i-1, j-1]+1
				else: 
					LCS_Buffer[i, j] = ti.math.max(LCS_Buffer[i-1, j], LCS_Buffer[i, j-1]) 
	
		# L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1] 
		return LCS_Buffer[TOTAL_ITEMS, TOTAL_ITEMS]
	
@ti.func
def knapsack_random_length_crossover(self, parent1: Individual, parent2: Individual, isl_ind: int):
	start = randint_isl(0, TOTAL_ITEMS-2, isl_ind)
	end = randint_isl(start, TOTAL_ITEMS-1, isl_ind)

	offspring1_genome = ti.Vector([-1 for _ in range(TOTAL_ITEMS)], dt=ti.i32)
	offspring2_genome = ti.Vector([-1 for _ in range(TOTAL_ITEMS)], dt=ti.i32)

	for i in range(start, end+1):
		offspring1_genome[i] = parent1.genome[i]
		offspring2_genome[i] = parent2.genome[i]

	parent1_pointer = (end + 1) % TOTAL_ITEMS
	parent2_pointer = (end + 1) % TOTAL_ITEMS
	# There's a more efficient way to do this I'm sure
	pointer = (end + 1) % TOTAL_ITEMS
	count = end-start+1
	while count < TOTAL_ITEMS:		
		offspring1_genome[pointer % TOTAL_ITEMS] = parent2.genome[parent2_pointer]
		count += 1
		pointer += 1
		parent2_pointer = (parent2_pointer + 1) % TOTAL_ITEMS

	pointer = (end + 1) % TOTAL_ITEMS
	count = end-start+1
	while count < TOTAL_ITEMS:						
		offspring2_genome[pointer % TOTAL_ITEMS] = parent1.genome[parent1_pointer]
		count += 1
		pointer += 1
		parent1_pointer = (parent1_pointer + 1) % TOTAL_ITEMS  
	
	return offspring1_genome, offspring2_genome