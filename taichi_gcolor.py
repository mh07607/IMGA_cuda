import taichi as ti
if __name__ == "__main__":
	ti.init(arch=ti.cpu, default_fp=ti.f64)

from taichi_rng import *
from taichi_gcolor_readdata import NUM_VERTICES, calculate_fitness, TYPE_GENOME, _generate_genome_isl, GRAPH, MAX_COLORS

LCS_Buffer = ti.field(dtype=ti.i32, shape=(NUM_VERTICES + 1, NUM_VERTICES + 1))

@ti.dataclass
class Individual:
	genome: TYPE_GENOME
	fitness: ti.f64
	genome_size: ti.i32
	
	# @ti.func
	# def initialize(self):
	# 	self.genome_size = NUM_VERTICES
	# 	self.genome = 
	
	@ti.func
	def initialize_isl(self, isl_ind):
		self.genome_size = NUM_VERTICES
		self.genome = _generate_genome_isl(isl_ind)
		self.fitness = calculate_fitness(self.genome)

	@ti.func
	def initialize_with_genome(self, genome):
		self.genome_size = NUM_VERTICES
		self.genome = genome
		self.fitness = calculate_fitness(self.genome)
			
	@ti.func
	def mutate_isl(self, max_colors, isl_ind) -> None:
		for vertex1 in range(NUM_VERTICES):
			for vertex2 in range(vertex1, NUM_VERTICES):
				if(GRAPH[vertex1, vertex2] == 1 and self.genome[vertex1] == self.genome[vertex2]):
					self.genome[vertex1] = randint_isl(1, max_colors, isl_ind)
		self.fitness = calculate_fitness(self.genome)		
		
	@ti.func
	def euclidean_distance(self, individual):
		return ti.math.distance(self.genome, individual.genome)

	@ti.func
	def hamming_distance(self, individual):
		difference = 0
		for i in range(NUM_VERTICES):
			if(self.genome[i] != individual.genome[i]):
				difference += 1
		return difference
	
	@ti.func
	def LCS(self, individual):		
		# declaring the array for storing the dp values 		
		"""Following steps build L[m + 1][n + 1] in bottom up fashion 
		Note: L[i][j] contains length of LCS of X[0..i-1] 
		and Y[0..j-1]"""
		for i in range(NUM_VERTICES + 1): 
			for j in range(NUM_VERTICES + 1): 
				if i == 0 or j == 0 : 
					LCS_Buffer[i, j] = 0
				elif self.genome[i-1] == individual.genome[j-1]: 
					LCS_Buffer[i, j] = LCS_Buffer[i-1, j-1]+1
				else: 
					LCS_Buffer[i, j] = ti.math.max(LCS_Buffer[i-1, j], LCS_Buffer[i, j-1]) 
	
		# L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1] 
		return LCS_Buffer[NUM_VERTICES, NUM_VERTICES]	

@ti.func
def gcolor_one_point_crossover(self, parent1, parent2, isl_ind): #one point crossover
	split_point = randint_isl(2, NUM_VERTICES - 2, isl_ind)

	# child1 = np.concatenate((parent1[:split_point], parent2[split_point:]))
	# child2 = np.concatenate((parent2[:split_point], parent1[split_point:]))
	offspring1_genome = ti.Vector([-1 for _ in range(NUM_VERTICES)], dt=ti.i32)
	offspring2_genome = ti.Vector([-1 for _ in range(NUM_VERTICES)], dt=ti.i32)

	for i in range(split_point):
		offspring1_genome[i] = parent1.genome[i]
		offspring2_genome[i] = parent2.genome[i]

	for i in range(split_point, NUM_VERTICES):
		offspring1_genome[i] = parent2.genome[i]
		offspring2_genome[i] = parent1.genome[i]

	return offspring1_genome, offspring2_genome

@ti.kernel
def testing_fitness():
	ind = Individual()
	ind.initialize_isl(0)
	print(ind.genome)
	print(ind.fitness)

if __name__ == "__main__":
	testing_fitness()