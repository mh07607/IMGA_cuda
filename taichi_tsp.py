import taichi as ti
if __name__ == "__main__":
    ti.init(arch=ti.cpu, default_fp=ti.f64)
from taichi_rng import *
    
from taichi_tsp_readdata import NUM_CITIES, distance, TYPE_GENOME, _generate_genome

# class TSP_Path(Individual):
# 	def __init__(self, genome):
# 		fitness = distance(genome)
# 		super().__init__(genome, fitness)
	
# 	def mutate(self) -> None:
# 		# mutation defined by reversing orders for now and to be changed
# 		# porposed algorithm would be to randomly swich 10 neighboring places such that it's neighbors have less distance as compared with the one
# 		rand_index1 = random.randint(0, len(self.genome)-1)
# 		rand_index2 = random.randint(0, len(self.genome)-1)

# 		self.genome[rand_index1], self.genome[rand_index2] = self.genome[rand_index2], self.genome[rand_index1]
# 		self.fitness = distance(self.genome)


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
    def mutate(self) -> None:
        rand_index1 = randint(0, self.genome_size)
        rand_index2 = randint(0, self.genome_size)
        self.genome[rand_index1], self.genome[rand_index2] = self.genome[rand_index2], self.genome[rand_index1]
        distance = distance(self.genome, self.genome_size)
        if rand_index1 != rand_index2:
            if ti.abs(distance - self.fitness) < 0.001:
                print("ERROR: mutation not working properly")
        self.fitness = distance
        
    

@ti.kernel
def test_kernel():
    individual = Individual()
    individual.initialize()
    # print(individual.genome)
    print("fitness: ", individual.fitness)
    individual.mutate()
    # print("genome: ", individual.genome.val)
    print("fitness: ", individual.fitness)
    print(individual.genome)
    
    
if __name__ == "__main__":

    test_kernel()
    