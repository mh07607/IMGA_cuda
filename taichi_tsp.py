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
def TSP_random_length_crossover(self, parent1: Individual, parent2: Individual) -> Individual_2_tuple:
    length = parent1.genome_size
    start = randint(1, length-3)
    end = randint(start, length-2)
    
    offspring1 = Individual()
    offspring2 = Individual()
    
    offspring1.genome = TYPE_GENOME([-1 for i in range(NUM_CITIES)])
    offspring2.genome = TYPE_GENOME([-1 for i in range(NUM_CITIES)])
    
    for i in range(start, end+1):
        offspring1.genome[i] = parent1.genome[i]
        offspring2.genome[i] = parent2.genome[i]
    
    pointer = end + 1
    parent1_pointer = end + 1
    parent2_pointer = end + 1
    
    while IN(offspring1.genome, NUM_CITIES, -1) == 1:
        if not (IN(offspring1.genome, NUM_CITIES, parent2.genome[parent2_pointer]) == 1):
            offspring1.genome[pointer % length] = parent2.genome[parent2_pointer]
            pointer += 1
        parent2_pointer = (parent2_pointer + 1) % length
        
    pointer = 0
    
    while (IN(offspring2.genome, NUM_CITIES, -1) == 1):
        if not (IN(offspring2.genome, NUM_CITIES, parent1.genome[parent1_pointer]) == 1):
            offspring2.genome[pointer % length] = parent1.genome[parent1_pointer]
            pointer += 1
        parent1_pointer = (parent1_pointer + 1) % length
        
    offspring1.fitness = distance(offspring1.genome, length)
    offspring2.fitness = distance(offspring2.genome, length)
    
    return Individual_2_tuple(first=offspring1, second=offspring2)

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
    parent1 = Individual()
    parent2 = Individual()
    parent1.initialize()
    parent2.initialize()
    offsprings = TSP_random_length_crossover(parent1, parent2)
    print("parent1: ", parent1.genome)
    print("parent2: ", parent2.genome)
    print("offspring1: ", offsprings.first.genome)
    print("offspring2: ", offsprings.second.genome)
    
if __name__ == "__main__":
    test_crossover()
    