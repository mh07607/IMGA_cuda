import taichi as ti
if __name__ == "__main__":
    ti.init(arch=ti.cpu, default_fp=ti.f64)
    
from taichi_rng import randint # similar to random.randint and random.sample
from taichi_tsp import Individual, TYPE_GENOME

POPULATION_SIZE = ti.field(dtype=ti.i32, shape=())
POPULATION_SIZE[None] = 100
POPULATION = Individual.field(shape=POPULATION_SIZE[None]*2)
POPULATION_POINTER = ti.field(dtype=ti.i32, shape=())
POPULATION_POINTER[None] = 0
SELECTION_RESULTS = Individual.field(shape=POPULATION_SIZE[None])

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
    indices = ti.Vector([i for i in range(POPULATION_SIZE[None])], dt=ti.i32)
    fitnesses = ti.Vector([POPULATION[i].fitness for i in range(POPULATION_SIZE[None])], dt=ti.f64)
    
    # Sort the array based on fitness values
    for i in range(POPULATION_SIZE[None]):
        for j in range(i + 1, POPULATION_SIZE[None]):
            if fitnesses[i] > fitnesses[j]:
                fitnesses[i], fitnesses[j] = fitnesses[j], fitnesses[i]
                indices[i], indices[j] = indices[j], indices[i]

    # Select the top num_selection elements and store them in SELECTION_RESULTS
    if res_opt == 0:
        for i in range(num_selections):
            SELECTION_RESULTS[i] = POPULATION[indices[i]]
    elif res_opt == 1:
        pass # any other storage option    
    
# @ti.func
# def random_selection(self, num_selections: ti.i32): # -> Feild of Indivdual Structs
#     survivors = ti.Vector.field(2, dtype=ti.f32, shape=(num_selections,))
#     for i in range(num_selections):
#         random_int = randint(0, len(self.population)-1)
#         survivors[i] = self.population[random_int]
#     return survivors

# @ti.func
# def binary_tournament_selection(self, num_selections: ti.i32):
#     result = ti.Vector.field(2, dtype=ti.f64, shape=(num_selections,))
#     for i in range(num_selections):
#         ind1, ind2 = sample(self.population, 2)
#         selected = ind1 if ind1.fitness < ind2.fitness else ind2
#         result[i] = selected
#     return result

########################## METHODS ##########################

# def get_average_and_best_individual(self) -> tuple[Individual, float]:
#     best_individual = self.population[0]
#     cumulative_fitness = 0
#     for individual in self.population:
#       if(individual.fitness < best_individual.fitness):
#         best_individual = individual
#       cumulative_fitness += individual.fitness
#     average_fitness = cumulative_fitness/len(self.population)
#     return best_individual, average_fitness


# def get_total_fitness(self) -> float:
# total_fitness = 0
# for individual in self.population:
#     total_fitness += individual.fitness
# return total_fitness

@ti.func
def get_avg_fitnes_n_best_indiv_index():
    best_index = 0
    cumulative_fitness = 0.0
    for i in range(POPULATION_SIZE[None]):
        individual = POPULATION[i]
        if individual.fitness < POPULATION[best_index].fitness:
            best_index = i
        cumulative_fitness += individual.fitness
    average_fitness = cumulative_fitness/POPULATION_SIZE[None]
    # BUG: note that best_index is an integer but returned as a float
    return ti.Vector([best_index, average_fitness], dt=ti.f64)

@ti.func
def get_total_fitness():
    total_fitness = 0.0
    for i in range(POPULATION_SIZE[None]):
        total_fitness += POPULATION[i].fitness
    return total_fitness

@ti.func
def initial_population_function():
    for i in range(POPULATION_SIZE[None]):
        POPULATION[i].initialize()
    POPULATION_POINTER[None] = POPULATION_SIZE[None]

########################## RUN ##########################

# def run_generation(self) -> None:
#     parents = self.parent_selection_function(self.num_offsprings)

#     # creating offspring
#     for k in range(0, self.num_offsprings-1, 2):
#       offspring1, offspring2 = self.cross_over_function(parents[k], parents[k+1])
#       rand_num1, rand_num2 = random.randint(0,100)/100, random.randint(0,100)/100
#       if rand_num1 <= self.mutation_rate:
#         offspring1.mutate()
#       if rand_num2 <= self.mutation_rate:
#         offspring2.mutate()
#       self.population.extend([offspring1, offspring2])

#     self.population = self.survivor_selection_function(self.population_size[None])

@ti.func
def run_generation(self):
    parents = self.parent_selection_function(self.num_offsprings)
    for k in range(0, self.num_offsprings-1, 2):
        offspring1, offspring2 = self.cross_over_function(parents[k], parents[k+1])
        rand_num1, rand_num2 = randint(0,100)/100, randint(0,100)/100
        if rand_num1 <= self.mutation_rate:
            offspring1.mutate()
        if rand_num2 <= self.mutation_rate:
            offspring2.mutate()
            
        POPULATION[POPULATION_POINTER[None]] = offspring1
        POPULATION_POINTER[None] += 1
        POPULATION[POPULATION_POINTER[None]] = offspring2
        POPULATION_POINTER[None] += 1

########################## TESTING ##########################
@ti.kernel
def test_truncation_selection():
    ti.loop_config(serialize=False)
    for i in range(POPULATION_SIZE[None]):
        POPULATION[i].initialize()
    for x in range(10):
        print(SELECTION_RESULTS[x].fitness)
    truncation_selection(10)
    for x in range(10):
        print(SELECTION_RESULTS[x].fitness)
    for x in range(POPULATION_SIZE[None]):
        print(POPULATION[x].fitness, end=', ')
        

@ti.kernel
def test_evol_algo():
    pass

    
if __name__ == "__main__":
    # ev = EvolutionaryAlgorithm()
    # ev.population = [Individual(), Individual()]
    # ev.population[0].initialize()
    # print(random_selection(ev, 2))
    # print(binary_tournament_selection(ev, 2))
    test_truncation_selection()
