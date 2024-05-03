import taichi as ti
if __name__ == "__main__":
    ti.init(arch=ti.cpu, default_fp=ti.f64)
    
from taichi_rng import randint # similar to random.randint and random.sample
from taichi_tsp import Individual, TYPE_GENOME, TSP_random_length_crossover

POPULATION_SIZE = 100

NUM_OFFSPRINGS = 2

POPULATION = Individual.field(shape=(POPULATION_SIZE + NUM_OFFSPRINGS))

PARENT_SELECTION = Individual.field(shape=NUM_OFFSPRINGS)

SELECTION_RESULTS = Individual.field(shape=POPULATION_SIZE)

@ti.dataclass
class EvolutionaryAlgorithm:
    # # These functions can be set using .methods when used in taichi
    # initial_population_function: FunctionType 
    # parent_selection_function: str,
    # survivor_selection_function: str,
    # cross_over_function: FunctionType, 
    mutation_rate: ti.f64
    num_offsprings: ti.i32
    population_size: ti.i32
    population_pointer: ti.i32

@ti.func
def truncation_selection(self, num_selections: ti.i32, res_opt: ti.i32):    
    if res_opt == 0: # parent selection
        # Temporary array to store indices and fitness values
        indices = ti.Vector([i for i in range(POPULATION_SIZE)], dt=ti.i32)
        fitnesses = ti.Vector([POPULATION[i].fitness for i in range(POPULATION_SIZE)], dt=ti.f64)
        
        # Sort the array based on fitness values
        for i in range(POPULATION_SIZE):
            for j in range(i + 1, POPULATION_SIZE):
                if fitnesses[i] > fitnesses[j]:
                    fitnesses[i], fitnesses[j] = fitnesses[j], fitnesses[i]
                    indices[i], indices[j] = indices[j], indices[i]
        for i in range(num_selections):
            PARENT_SELECTION[i] = POPULATION[indices[i]]
    elif res_opt == 1: # survivor selection
        # Temporary array to store indices and fitness values
        indices = ti.Vector([i for i in range(POPULATION_SIZE + NUM_OFFSPRINGS)], dt=ti.i32)
        fitnesses = ti.Vector([POPULATION[i].fitness for i in range(POPULATION_SIZE + NUM_OFFSPRINGS)], dt=ti.f64)
        # Sort the array based on fitness values
        for i in range(POPULATION_SIZE + NUM_OFFSPRINGS):
            for j in range(i + 1, POPULATION_SIZE + NUM_OFFSPRINGS):
                if fitnesses[i] > fitnesses[j]:
                    fitnesses[i], fitnesses[j] = fitnesses[j], fitnesses[i]
                    indices[i], indices[j] = indices[j], indices[i]

        for i in range(num_selections):
            POPULATION[i] = POPULATION[indices[i]]

########################## METHODS ##########################

@ti.func
def get_avg_fitnes_n_best_indiv_index():
    best_index = 0
    cumulative_fitness = 0.0
    for i in range(POPULATION_SIZE):
        individual = POPULATION[i]
        if individual.fitness < POPULATION[best_index].fitness:
            best_index = i
        cumulative_fitness += individual.fitness
    average_fitness = cumulative_fitness/POPULATION_SIZE
    # BUG: note that best_index is an integer but returned as a float
    return ti.Vector([best_index, average_fitness], dt=ti.f64)

@ti.func
def get_total_fitness():
    total_fitness = 0.0
    for i in range(POPULATION_SIZE):
        total_fitness += POPULATION[i].fitness
    return total_fitness

@ti.func
def initial_population_function():
    for i in range(POPULATION_SIZE):
        POPULATION[i].initialize()

########################## RUN ##########################

@ti.func
def run_generation(self):
    self.parent_selection_function(self.num_offsprings, 0)
    for k in range(0, self.num_offsprings-1):
        if k % 2 == 1:
            continue
        # print(PARENT_SELECTION[k].genome, PARENT_SELECTION[k+1].genome)
        offspring1_genome, offspring2_genome = self.cross_over_function(PARENT_SELECTION[k], PARENT_SELECTION[k+1])
        offspring1 = Individual()
        offspring1.initialize_with_genome(offspring1_genome)
        offspring2 = Individual()
        offspring2.initialize_with_genome(offspring2_genome)

        rand_num1, rand_num2 = randint(0,100)/100, randint(0,100)/100
        if rand_num1 <= self.mutation_rate:
            offspring1.mutate()
        if rand_num2 <= self.mutation_rate:
            offspring2.mutate()
            
        POPULATION[POPULATION_SIZE+k] = offspring1
        POPULATION[POPULATION_SIZE+k+1] = offspring2
        
    self.survivor_selection_function(POPULATION_SIZE, 1)
    

@ti.kernel
def run(EA: EvolutionaryAlgorithm, num_iterations: ti.i32, num_generations: ti.i32) -> ti.i32:
        initial_population_function()
        # sum_avg_fitness = 0.0        
        ti.loop_config(serialize=True)
        best_index = 0
        for i in range(num_generations):
            EA.run_generation()
            # best_index is always 0 so we don't need this function
            best_index, avg_fitness = get_avg_fitnes_n_best_indiv_index()
            best_index = ti.i32(best_index)
            print("generation: ", i)        
            print("best_individual: ", best_index, "fitness: ", POPULATION[best_index].fitness)                    
        return best_index
        
        


########################## TESTING ##########################
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
    EvolutionaryAlgorithm.methods = {
        'cross_over_function': TSP_random_length_crossover,
        "parent_selection_function": truncation_selection,
        "survivor_selection_function": truncation_selection,
        'run_generation': run_generation,
    }
    EA = EvolutionaryAlgorithm(mutation_rate=0.5, num_offsprings=20)
    run(EA, 10, 1000)
