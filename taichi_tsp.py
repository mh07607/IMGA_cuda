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
    
    
if __name__ == "__main__":
    test_kernel()
    