import taichi as ti
if __name__ == "__main__":
    ti.init(arch=ti.cpu, default_fp=ti.f64)
    
from taichi_rng import randint # similar to random.randint and random.sample
from taichi_tsp import Individual, TYPE_GENOME

POPULATION_SIZE = 100
POPULATION = Individual.field(shape=POPULATION_SIZE)
SELECTION_RESULTS = Individual.field(shape=POPULATION_SIZE)

@ti.dataclass
class EvolutionaryAlgorithm:
    # # These functions can be set using .methods when used in taichi
    # initial_population_function: FunctionType 
    # parent_selection_function: str,
    # survivor_selection_function: str,
    # cross_over_function: FunctionType, 
    mutation_rate: float
    num_offsprings: int
    

@ti.func
def truncation_selection(num_selections: ti.i32, res_opt: ti.i32 = 0):
    # Temporary array to store indices and fitness values
    indices = ti.Vector([i for i in range(POPULATION_SIZE)], dt=ti.i32)
    fitnesses = ti.Vector([POPULATION[i].fitness for i in range(POPULATION_SIZE)], dt=ti.f64)
    
    # Sort the array based on fitness values
    for i in range(POPULATION_SIZE):
        for j in range(i + 1, POPULATION_SIZE):
            if fitnesses[i] > fitnesses[j]:
                fitnesses[i], fitnesses[j] = fitnesses[j], fitnesses[i]
                indices[i], indices[j] = indices[j], indices[i]

    # Select the top num_selection elements and store them in SELECTION_RESULTS
    if res_opt == 0:
        for i in range(num_selections):
            SELECTION_RESULTS[i] = POPULATION[indices[i]]
    elif res_opt == 1:
        pass # any other storage option
    
@ti.func
def random_selection(self, num_selections: ti.i32): # -> Feild of Indivdual Structs
    survivors = ti.Vector.field(2, dtype=ti.f32, shape=(num_selections,))
    for i in range(num_selections):
        random_int = randint(0, len(self.population)-1)
        survivors[i] = self.population[random_int]
    return survivors
# @ti.func
# def binary_tournament_selection(self, num_selections: ti.i32):
#     result = ti.Vector.field(2, dtype=ti.f64, shape=(num_selections,))
#     for i in range(num_selections):
#         ind1, ind2 = sample(self.population, 2)
#         selected = ind1 if ind1.fitness < ind2.fitness else ind2
#         result[i] = selected
#     return result
    

@ti.kernel
def test_truncation_selection():
    ti.loop_config(serialize=False)
    for i in range(POPULATION_SIZE):
        POPULATION[i].initialize()
    for x in range(10):
        print(SELECTION_RESULTS[x].fitness)
    truncation_selection(10)
    for x in range(10):
        print(SELECTION_RESULTS[x].fitness)
    for x in range(POPULATION_SIZE):
        print(POPULATION[x].fitness, end=', ')
    
if __name__ == "__main__":
    # ev = EvolutionaryAlgorithm()
    # ev.population = [Individual(), Individual()]
    # ev.population[0].initialize()
    # print(random_selection(ev, 2))
    # print(binary_tournament_selection(ev, 2))
    test_truncation_selection()
