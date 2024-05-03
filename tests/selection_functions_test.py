import numpy as np

NUM_OFFSPRINGS = 2
POPULATION_SIZE = 8

ISL_POPULATIONS = np.arange(1, POPULATION_SIZE + NUM_OFFSPRINGS + 1, 1).reshape((1, POPULATION_SIZE + NUM_OFFSPRINGS))
ISL_SELECTION_RESULTS = np.zeros(POPULATION_SIZE + NUM_OFFSPRINGS).reshape((1, POPULATION_SIZE + NUM_OFFSPRINGS))
ISL_PARENT_SELECTIONS = np.zeros(NUM_OFFSPRINGS).reshape((1, NUM_OFFSPRINGS))


def random_selection(isl_ind, res_opt):    
    if(res_opt == 0):
        for i in range(NUM_OFFSPRINGS):
            rand_index = np.random.randint(0, POPULATION_SIZE)
            ISL_PARENT_SELECTIONS[isl_ind, i] = ISL_POPULATIONS[isl_ind, rand_index]
    elif(res_opt == 1):
            indices = [i for i in range(POPULATION_SIZE + NUM_OFFSPRINGS)]
            count = 0
            for i in range(POPULATION_SIZE + NUM_OFFSPRINGS):            
                num_indices_left = POPULATION_SIZE + NUM_OFFSPRINGS - count                
                rand_index = np.random.randint(0, num_indices_left)
                ISL_SELECTION_RESULTS[isl_ind, i] = ISL_POPULATIONS[isl_ind, indices[rand_index]]
                for j in range(rand_index, num_indices_left-1):
                    indices[j] = indices[j+1] 
                count += 1           

            for i in range(POPULATION_SIZE):
                ISL_POPULATIONS[isl_ind, i] = ISL_SELECTION_RESULTS[isl_ind, i]


def rank_based_selection(isl_ind, res_opt):
    pass

def fitness_proportional_selection(isl_ind, res_opt):
    if(res_opt == 0):
        population_proportions = [0.0 for _ in range(POPULATION_SIZE)]
        cumulative_fitness = 0.0            
        for i in range(POPULATION_SIZE):
            individual = ISL_POPULATIONS[isl_ind, i]        
            cumulative_fitness += (1/individual) # change to fitness
            population_proportions[i] = cumulative_fitness        
        total_fitness = cumulative_fitness            
        for i in range(NUM_OFFSPRINGS):
            random_float = np.random.uniform(0, total_fitness)
            for j in range(POPULATION_SIZE-1):
                if population_proportions[j+1] > random_float:
                    ISL_PARENT_SELECTIONS[isl_ind, i] = ISL_POPULATIONS[isl_ind, j - 1]
                    break
    elif(res_opt == 1):
        population_proportions = [0.0 for _ in range(POPULATION_SIZE + NUM_OFFSPRINGS)]
        cumulative_fitness = 0.0            
        for i in range(POPULATION_SIZE + NUM_OFFSPRINGS):
            individual = ISL_POPULATIONS[isl_ind, i] 
            ISL_SELECTION_RESULTS[isl_ind, i] = individual
            cumulative_fitness += (1/individual) # change to fitness
            population_proportions[i] = cumulative_fitness        
        total_fitness = cumulative_fitness            
        for i in range(POPULATION_SIZE):
            random_float = np.random.uniform(0, total_fitness)
            for j in range(POPULATION_SIZE):
                if population_proportions[j+1] > random_float:
                    ISL_POPULATIONS[isl_ind, i] = ISL_SELECTION_RESULTS[isl_ind, j - 1]
                    break
        


def test_fitness_selection():
    print(ISL_POPULATIONS)
    print(ISL_PARENT_SELECTIONS)
    fitness_proportional_selection(isl_ind=0, res_opt=0)
    print(ISL_PARENT_SELECTIONS)    
    fitness_proportional_selection(isl_ind=0, res_opt=1)
    print(ISL_SELECTION_RESULTS)
    print(ISL_POPULATIONS)


def test_random_selection():
    print(ISL_PARENT_SELECTIONS)
    random_selection(isl_ind=0, res_opt=0)
    print(ISL_PARENT_SELECTIONS)
    print(ISL_POPULATIONS)
    random_selection(isl_ind=0, res_opt=1)
    print(ISL_SELECTION_RESULTS)
    print(ISL_POPULATIONS)


# test_random_selection()
test_fitness_selection()