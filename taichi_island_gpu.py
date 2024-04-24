import taichi as ti
if __name__ == "__main__":
    ti.init(arch=ti.gpu, default_fp=ti.f64)
    
from taichi_rng import randint # similar to random.randint and random.sample
from taichi_tsp import Individual, TYPE_GENOME, TSP_random_length_crossover

# POPULATION_SIZE = ti.field(dtype=ti.i32, shape=())
POPULATION_SIZE = 100
NUM_ISLANDS = 10

# NUM_OFFSPRINGS = ti.field(dtype=ti.i32, shape=())
NUM_OFFSPRINGS = 20

ISL_POPULATIONS = Individual.field(shape=(NUM_ISLANDS, POPULATION_SIZE + NUM_OFFSPRINGS))

ISL_PARENT_SELECTIONS = Individual.field(shape=(NUM_ISLANDS, NUM_OFFSPRINGS))

ISL_SELECTION_RESULTS = Individual.field(shape=(NUM_ISLANDS, POPULATION_SIZE))

BEST_INDICES = ti.field(dtype=ti.i32, shape=(NUM_ISLANDS))

@ti.dataclass
class EvolutionaryAlgorithm:
    # # These functions can be set using .methods when used in taichi
    # initial_population_function: FunctionType 
    # parent_selection_function: str,
    # survivor_selection_function: str,
    # cross_over_function: FunctionType, 
    mutation_rate: ti.f64

@ti.func
def truncation_selection(self, isl_ind: ti.i32, num_selections: ti.i32, res_opt: ti.i32):    
    if res_opt == 0: # parent selection
        # Temporary array to store indices and fitness values
        indices = ti.Vector([i for i in range(POPULATION_SIZE)], dt=ti.i32)
        fitnesses = ti.Vector([ISL_POPULATIONS[isl_ind, i].fitness for i in range(POPULATION_SIZE)], dt=ti.f64)
        
        # Sort the array based on fitness values
        for i in range(POPULATION_SIZE):
            for j in range(i + 1, POPULATION_SIZE):
                if fitnesses[i] > fitnesses[j]:
                    fitnesses[i], fitnesses[j] = fitnesses[j], fitnesses[i]
                    indices[i], indices[j] = indices[j], indices[i]
        for i in range(num_selections):
            ISL_PARENT_SELECTIONS[isl_ind, i] = ISL_POPULATIONS[isl_ind, indices[i]]
            
    elif res_opt == 1: # survivor selection
        # Temporary array to store indices and fitness values
        indices = ti.Vector([i for i in range(POPULATION_SIZE + NUM_OFFSPRINGS)], dt=ti.i32)
        fitnesses = ti.Vector([ISL_POPULATIONS[isl_ind, i].fitness for i in range(POPULATION_SIZE + NUM_OFFSPRINGS)], dt=ti.f64)
        # Sort the array based on fitness values
        for i in range(POPULATION_SIZE + NUM_OFFSPRINGS):
            for j in range(i + 1, POPULATION_SIZE + NUM_OFFSPRINGS):
                if fitnesses[i] > fitnesses[j]:
                    fitnesses[i], fitnesses[j] = fitnesses[j], fitnesses[i]
                    indices[i], indices[j] = indices[j], indices[i]

        for i in range(num_selections):
            ISL_POPULATIONS[isl_ind, i] = ISL_POPULATIONS[isl_ind, indices[i]]



########################## METHODS ##########################

@ti.func
def get_avg_fitnes_n_best_indiv_index(isl_ind: ti.i32):
    best_index = 0
    cumulative_fitness = 0.0
    for i in range(POPULATION_SIZE):
        individual = ISL_POPULATIONS[isl_ind, i]
        if individual.fitness < ISL_POPULATIONS[isl_ind, best_index].fitness:
            best_index = i
        cumulative_fitness += individual.fitness
    average_fitness = cumulative_fitness/POPULATION_SIZE
    # BUG: note that best_index is an integer but returned as a float
    return ti.Vector([best_index, average_fitness], dt=ti.f64)

@ti.func
def get_total_fitness(isl_ind: ti.i32):
    total_fitness = 0.0
    for i in range(POPULATION_SIZE):
        total_fitness += ISL_POPULATIONS[isl_ind, i].fitness
    return total_fitness

@ti.func
def initial_population_function(isl_ind: ti.i32):
    for i in range(POPULATION_SIZE):
        ISL_POPULATIONS[isl_ind, i].initialize()

########################## RUN ##########################

        
'''
Migration strategies will go here
- Ring migration, Hamming distance similarity, LCS
'''
@ti.func
def ring_migration(isl_ind):
    next_island = (isl_ind + 1) % NUM_ISLANDS
    # select a random position to replace
    replace_index = randint(0, POPULATION_SIZE-1)
    ISL_POPULATIONS[next_island, replace_index] = ISL_POPULATIONS[isl_ind, BEST_INDICES[isl_ind]]

        
@ti.func
def i_run_generation(self, isl_ind: ti.i32):
    '''
    NOTES
    - All selection functions will need to know which island for selections i.e. island index
    - We need to return best_individual index in each selection function for migration purpose
    - Will migrants add to the population of each island? (We can place it randomly in the island 
    population for now but in general population increases)
    - Need to keep track of the best individual amongst all islands, we can do this using the
    individual indices array
    
    - We need to make functions for different migration strategies
    - Migrate after how many generations?
    - Each island can have separate configuration, can it be adaptive?
    - Isn't it better practice to pass the islands population itself in the selection functions
    instead of the index. The only pitfall could be if the values in place are not changed but 
    I think they will be
    '''
    self.parent_selection_function(isl_ind, NUM_OFFSPRINGS, 0)
    for k in range(0, NUM_OFFSPRINGS-1):
        if k % 2 == 1:
            continue
        # BUG WARNING: printing genome causes issues for later usage in the scope. I think
        # this is since print is performed in python scope and not taichi scope which causes
        # this sort of undeterministic behaviour
        # print(PARENT_SELECTION[k].genome, PARENT_SELECTION[k+1].genome)
        offspring1_genome, offspring2_genome = self.cross_over_function(ISL_PARENT_SELECTIONS[isl_ind, k], ISL_PARENT_SELECTIONS[isl_ind, k+1])
        offspring1 = Individual()
        offspring1.initialize_with_genome(offspring1_genome)
        offspring2 = Individual()
        offspring2.initialize_with_genome(offspring2_genome)

        rand_num1, rand_num2 = randint(0,100)/100, randint(0,100)/100
        if rand_num1 <= self.mutation_rate:
            offspring1.mutate()
        if rand_num2 <= self.mutation_rate:
            offspring2.mutate()
            
        ISL_POPULATIONS[isl_ind, POPULATION_SIZE+k] = offspring1
        ISL_POPULATIONS[isl_ind, POPULATION_SIZE+k+1] = offspring2
        
    best_index = self.survivor_selection_function(isl_ind, POPULATION_SIZE, 1)
    return best_index
        
@ti.kernel
def run_islands(EA: EvolutionaryAlgorithm, num_islands: ti.i32, num_iterations: ti.i32, num_generations: ti.i32):
    for isl_ind in range(num_islands):
        initial_population_function(isl_ind)
        print("initial population for island", isl_ind, "is", get_total_fitness(isl_ind))
        best_index = 0
        for i in range(num_generations):
            # JAADU
            # if (i)%50 == 0:
            #     ti.simt.block.sync()
            #     ring_migration(isl_ind)
            EA.run_generation(isl_ind)
            # best_index is always 0 so we don't need this function
            best_index, avg_fitness = get_avg_fitnes_n_best_indiv_index(isl_ind)
            best_index = ti.i32(best_index)
            
        BEST_INDICES[isl_ind] = best_index
        print("best index for island", isl_ind, "is", best_index)

    
        
    
    
########################## TESTING ##########################
# @ti.kernel
# def test_truncation_selection():
#     ti.loop_config(serialize=False)
#     for i in range(POPULATION_SIZE):
#         POPULATION[i].initialize()
#     for x in range(10):
#         print(SELECTION_RESULTS[x].fitness)
#     truncation_selection(10)
#     for x in range(10):
#         print(SELECTION_RESULTS[x].fitness)
#     for x in range(POPULATION_SIZE):
#         print(POPULATION[x].fitness, end=', ')
        


if __name__ == "__main__":
    EvolutionaryAlgorithm.methods = {
        'cross_over_function': TSP_random_length_crossover,
        "parent_selection_function": truncation_selection,
        "survivor_selection_function": truncation_selection,
        'run_generation': i_run_generation,
    }
    EA = EvolutionaryAlgorithm(mutation_rate=0.5)
    run_islands(EA, NUM_ISLANDS, 10, 1000)
    for isl_ind in range(NUM_ISLANDS):
        print(BEST_INDICES[isl_ind])
        print(ISL_POPULATIONS[isl_ind, BEST_INDICES[isl_ind]].fitness)
