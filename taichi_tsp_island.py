import taichi as ti
if __name__ == "__main__":
	ti.init(arch=ti.gpu, default_fp=ti.f64)
	
from taichi_rng import randint, randint_isl, randfloat_isl # similar to random.randint and random.sample
from taichi_tsp import Individual, TYPE_GENOME, TSP_random_length_crossover
import numpy as np
import matplotlib.pyplot as plt
import time
import math

# POPULATION_SIZE = ti.field(dtype=ti.i32, shape=())
POPULATION_SIZE = 100
NUM_ISLANDS = 64

# NUM_OFFSPRINGS = ti.field(dtype=ti.i32, shape=())
NUM_OFFSPRINGS = 10

ISL_POPULATIONS = Individual.field(shape=(NUM_ISLANDS, POPULATION_SIZE + NUM_OFFSPRINGS))

ISL_PARENT_SELECTIONS = Individual.field(shape=(NUM_ISLANDS, NUM_OFFSPRINGS))

ISL_SELECTION_RESULTS = Individual.field(shape=(NUM_ISLANDS, POPULATION_SIZE + NUM_OFFSPRINGS))

BEST_INDICES = ti.field(dtype=ti.i32, shape=(NUM_ISLANDS))
BEST_INDICES_GENERATION = Individual.field(shape=(50, NUM_ISLANDS))

@ti.dataclass
class EvolutionaryAlgorithm:
	# # These functions can be set using .methods when used in taichi
	# initial_population_function: FunctionType 
	# parent_selection_function: str,
	# survivor_selection_function: str,
	# cross_over_function: FunctionType, 
	mutation_rate: ti.f64

ISLANDS = EvolutionaryAlgorithm.field(shape=(NUM_ISLANDS,))

'''
TO-DO
1. We should return indices from this function instead of making a separate buffer
at least in the case of parent selection. In this way we can just return a Vector
and not have to use a global memory access. In case of survivor selection, we can just
simply return an empty Vector along with 
2. We should replace bubble sort here with merge-sort.
3. We should get rid of num_selections and just use POPULATION_SIZE and NUM_OFFSPRINGS as parameters
'''
@ti.func
def truncation_selection(self, isl_ind: ti.i32, res_opt: ti.i32):    
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
		
		# Selecting parents
		for i in range(NUM_OFFSPRINGS):
			ISL_PARENT_SELECTIONS[isl_ind, i] = ISL_POPULATIONS[isl_ind, indices[i]]
			
	elif res_opt == 1: # survivor selection
		# Temporary array to store indices and fitness values
		# indices = ti.Vector([i for i in range(POPULATION_SIZE + NUM_OFFSPRINGS)], dt=ti.i32)
		# fitnesses = ti.Vector([ISL_POPULATIONS[isl_ind, i].fitness for i in range(POPULATION_SIZE + NUM_OFFSPRINGS)], dt=ti.f64)

		# Sort the array based on fitness values, nothing else is required
		for i in range(POPULATION_SIZE + NUM_OFFSPRINGS):
			for j in range(i + 1, POPULATION_SIZE + NUM_OFFSPRINGS):
				if ISL_POPULATIONS[isl_ind, i].fitness > ISL_POPULATIONS[isl_ind, j].fitness:
					ISL_POPULATIONS[isl_ind, i], ISL_POPULATIONS[isl_ind, j] = ISL_POPULATIONS[isl_ind, j], ISL_POPULATIONS[isl_ind, i]                

		# for i in range(num_selections):
		#     # BUG: this code is problematic, ISL_POPULATIONS is being overwritten 
		#     # before all selections are made
		#     ISL_POPULATIONS[isl_ind, i] = ISL_POPULATIONS[isl_ind, indices[i]]


@ti.func
def binary_tournament_selection(self, isl_ind: ti.i32, res_opt: ti.i32):
	if res_opt == 0: # parent selection
		for i in range(NUM_OFFSPRINGS):
			ind1_idx = randint(0, POPULATION_SIZE)
			ind2_idx = randint(0, POPULATION_SIZE)
			ind1 = ISL_POPULATIONS[isl_ind, ind1_idx]
			ind2 = ISL_POPULATIONS[isl_ind, ind2_idx]
			selected = ind2
			if(ind1.fitness < ind2.fitness):
				selected = ind1						
			ISL_PARENT_SELECTIONS[isl_ind, i] = selected
	elif res_opt == 1: # survivor selection
		for i in range(POPULATION_SIZE + NUM_OFFSPRINGS):
			ind1_idx = randint(0, POPULATION_SIZE)
			ind2_idx = randint(0, POPULATION_SIZE)
			ind1 = ISL_POPULATIONS[isl_ind, ind1_idx]
			ind2 = ISL_POPULATIONS[isl_ind, ind2_idx]
			selected = ind2
			if(ind1.fitness < ind2.fitness):
				selected = ind1		
			ISL_SELECTION_RESULTS[isl_ind, i] = selected

		for i in range(POPULATION_SIZE):			
			ISL_POPULATIONS[isl_ind, i] = ISL_SELECTION_RESULTS[isl_ind, i]
		
@ti.func
def fitness_proportional_selection(isl_ind, res_opt):
    if(res_opt == 0):
        population_proportions = ti.Vector([0.0 for _ in range(POPULATION_SIZE)])
        cumulative_fitness = 0.0            
        for i in range(POPULATION_SIZE):
            individual = ISL_POPULATIONS[isl_ind, i]        
            cumulative_fitness += (1/individual) # change to fitness
            population_proportions[i] = cumulative_fitness        
        total_fitness = cumulative_fitness            
        for i in range(NUM_OFFSPRINGS):
            random_float = randfloat_isl(0, total_fitness, isl_ind)
            for j in range(POPULATION_SIZE-1):
                if population_proportions[j+1] > random_float:
                    ISL_PARENT_SELECTIONS[isl_ind, i] = ISL_POPULATIONS[isl_ind, j - 1]
                    break
    elif(res_opt == 1):
        population_proportions = ti.Vector([0.0 for _ in range(POPULATION_SIZE + NUM_OFFSPRINGS)])
        cumulative_fitness = 0.0            
        for i in range(POPULATION_SIZE + NUM_OFFSPRINGS):
            individual = ISL_POPULATIONS[isl_ind, i] 
            ISL_SELECTION_RESULTS[isl_ind, i] = individual
            cumulative_fitness += (1/individual) # change to fitness
            population_proportions[i] = cumulative_fitness        
        total_fitness = cumulative_fitness            
        for i in range(POPULATION_SIZE):
            random_float = randfloat_isl(0, total_fitness)
            for j in range(POPULATION_SIZE):
                if population_proportions[j+1] > random_float:
                    ISL_POPULATIONS[isl_ind, i] = ISL_SELECTION_RESULTS[isl_ind, j - 1]
                    break


@ti.func
def random_selection(self, isl_ind: ti.i32, res_opt: ti.i32):    
	if(res_opt == 0):
		for i in range(NUM_OFFSPRINGS):
			rand_index = randint(0, POPULATION_SIZE)
			ISL_PARENT_SELECTIONS[isl_ind, i] = ISL_POPULATIONS[isl_ind, rand_index]
	elif(res_opt == 1):
			indices = ti.Vector([i for i in range(POPULATION_SIZE + NUM_OFFSPRINGS)], dt=ti.i32)
			count = 0
			for i in range(POPULATION_SIZE + NUM_OFFSPRINGS - 1):                
				num_indices_left = POPULATION_SIZE + NUM_OFFSPRINGS - count
				rand_index = randint(0, num_indices_left)
				ISL_SELECTION_RESULTS[isl_ind, i] = ISL_POPULATIONS[isl_ind, indices[rand_index]]
				for j in range(rand_index, num_indices_left-1):
					indices[j] = indices[j+1]
				count += 1

			for i in range(POPULATION_SIZE):
				ISL_POPULATIONS[isl_ind, i] = ISL_SELECTION_RESULTS[isl_ind, i]
		
@ti.func
def rank_selection(self, isl_ind: ti.i32, res_opt: ti.i32):    
	pass



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
		ISL_POPULATIONS[isl_ind, i].initialize_isl(isl_ind)

########################## RUN ##########################

		
'''
Migration strategies will go here
- Ring migration, Hamming distance similarity, LCS
'''

# # LCS adapted from GeeksforGeeks
# @ti.func
# def LCS(X, Y): 
# 	# find the length of the strings 
# 	m = len(X) 
# 	n = len(Y) 
 
# 	# declaring the array for storing the dp values 
# 	L = [[None]*(n + 1) for i in range(m + 1)] 
 
# 	"""Following steps build L[m + 1][n + 1] in bottom up fashion 
# 	Note: L[i][j] contains length of LCS of X[0..i-1] 
# 	and Y[0..j-1]"""
# 	for i in range(m + 1): 
# 		for j in range(n + 1): 
# 			if i == 0 or j == 0 : 
# 				L[i][j] = 0
# 			elif X[i-1] == Y[j-1]: 
# 				L[i][j] = L[i-1][j-1]+1
# 			else: 
# 				L[i][j] = max(L[i-1][j], L[i][j-1]) 
 
# 	# L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1] 
# 	return L[m][n] 

'''
TO-DO
1. In all migration strategies, We need to make a buffer of best individuals
separately otherwise it may get overwritten by migrated individuals
'''
@ti.func
def hamming_based_migration(self):
	'''
	  Migration based on lowest 
	'''
	for isl_ind in range(NUM_ISLANDS):		
		# select a random position to replace
		isl_best_indiv = ISL_POPULATIONS[isl_ind, BEST_INDICES[isl_ind]]
		least_distance_index = -1
		least_distance = ti.math.inf
		for other_ind in range(NUM_ISLANDS):
			if(other_ind == isl_ind):
				continue
			distance_bw_best = isl_best_indiv.hamming_distance(ISL_POPULATIONS[other_ind, BEST_INDICES[other_ind]])
			if(distance_bw_best < least_distance):
				least_distance = distance_bw_best
				least_distance_index = other_ind
		replace_index = randint_isl(0, POPULATION_SIZE-1, isl_ind)
		ISL_POPULATIONS[least_distance_index, replace_index] = isl_best_indiv


@ti.func
def LCS_based_migration(self):
	for isl_ind in range(NUM_ISLANDS):		
		# select a random position to replace
		isl_best_indiv = ISL_POPULATIONS[isl_ind, BEST_INDICES[isl_ind]]
		best_similarity_index = -1
		best_similarity = 0
		for other_ind in range(NUM_ISLANDS):
			if(other_ind == isl_ind):
				continue
			similarity_bw_best = isl_best_indiv.LCS(ISL_POPULATIONS[other_ind, BEST_INDICES[other_ind]])
			if(similarity_bw_best > best_similarity):
				best_similarity = similarity_bw_best
				best_similarity_index = other_ind
		replace_index = randint_isl(0, POPULATION_SIZE-1, isl_ind)
		ISL_POPULATIONS[best_similarity_index, replace_index] = isl_best_indiv

@ti.func
def ring_migration(self):
	for isl_ind in range(NUM_ISLANDS):
		next_island = (isl_ind + 1) % NUM_ISLANDS
		# select a random position to replace
		replace_index = randint_isl(0, POPULATION_SIZE-1, isl_ind)
		ISL_POPULATIONS[next_island, replace_index] = ISL_POPULATIONS[isl_ind, BEST_INDICES[isl_ind]]

		
@ti.func
def i_run_generation(self, isl_ind: ti.i32):
	'''
	NOTES
	- All selection functions will need to know which island for selections i.e. island index
	- We need to return best_individual index in each selection function for migration purpose
	- Will migrants add to the population of each island? (We can place it randomly in the island 
	population for now but waise population increases)
	- Need to keep track of the best individual amongst all islands, we can do this using the
	individual indices array
	
	- We need to make functions for different migration strategies
	- Migrate after variable n generations?
	- Each island can have separate configuration, can it be adaptive?
	- Isn't it better practice to pass the islands population itself in the selection functions
	instead of the index. The only pitfall could be if the values in place are not changed but 
	I think they will be
	'''
	self.parent_selection_function(isl_ind, 0)
	for k in range(0, NUM_OFFSPRINGS-1):
		if k % 2 == 1:
			continue
		# BUG WARNING: printing genome causes issues for later usage in the scope. I think
		# this is since print is performed in python scope and not taichi scope which causes
		# this sort of undeterministic behaviour
		# print(PARENT_SELECTION[k].genome, PARENT_SELECTION[k+1].genome)
		offspring1_genome, offspring2_genome = self.cross_over_function(ISL_PARENT_SELECTIONS[isl_ind, k], ISL_PARENT_SELECTIONS[isl_ind, k+1], isl_ind)
		offspring1 = Individual()
		offspring1.initialize_with_genome(offspring1_genome)
		offspring2 = Individual()
		offspring2.initialize_with_genome(offspring2_genome)

		rand_num1, rand_num2 = randint_isl(0,100,isl_ind)/100, randint_isl(0,100, isl_ind)/100
		if rand_num1 <= self.mutation_rate:
			offspring1.mutate_isl(isl_ind)
		if rand_num2 <= self.mutation_rate:
			offspring2.mutate_isl(isl_ind)
			
		ISL_POPULATIONS[isl_ind, POPULATION_SIZE+k] = offspring1
		ISL_POPULATIONS[isl_ind, POPULATION_SIZE+k+1] = offspring2
		
	self.survivor_selection_function(isl_ind, 1)
		
@ti.kernel
def run_islands(EA: EvolutionaryAlgorithm, num_islands: ti.i32, migration_step: ti.i32, num_generations: ti.i32):
	# ti.block_local(ISL_POPULATIONS)
	# ti.loop_config(block_dim=NUM_ISLANDS)
	for isl_ind in range(num_islands):
		initial_population_function(isl_ind)
		best_index = 0
		for i in range(num_generations):
			# JAADU
			if (i + 1)% migration_step == 0:
				ti.simt.block.sync()
				if(isl_ind == 0):
					EA.migration()
				ti.simt.block.sync()
			EA.run_generation(isl_ind)
			# best_index is always 0 so we don't need this function
			best_index, avg_fitness = get_avg_fitnes_n_best_indiv_index(isl_ind)
			best_index = ti.i32(best_index)
			
			BEST_INDICES_GENERATION[i, isl_ind] = ISL_POPULATIONS[isl_ind, best_index]
		BEST_INDICES[isl_ind] = best_index		

@ti.kernel
def run_islands_cpu(EA: EvolutionaryAlgorithm, num_islands: ti.i32, migration_step: ti.i32, num_generations: ti.i32):
	for isl_ind in range(num_islands):
		initial_population_function(isl_ind)
		best_index = 0
		for i in range(num_generations):
			# JAADU
			if (i + 1) % migration_step == 0:                
				EA.migration()
			EA.run_generation(isl_ind)
			# best_index is always 0 so we don't need this function
			best_index, avg_fitness = get_avg_fitnes_n_best_indiv_index(isl_ind)
			best_index = ti.i32(best_index)
			
		BEST_INDICES[isl_ind] = best_index		

	
	
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
		"parent_selection_function": fitness_proportional_selection,
		"survivor_selection_function": truncation_selection,
		'run_generation': i_run_generation,
		"migration": ring_migration
	}	
	EA = EvolutionaryAlgorithm(mutation_rate=0.5)
	starting_time = time.time()	
	run_islands(EA, NUM_ISLANDS, 5, 50)
	for isl_ind in range(NUM_ISLANDS):
		print(ISL_POPULATIONS[isl_ind, BEST_INDICES[isl_ind]].fitness)
	ending_time = time.time() - starting_time
	print("Time taken", ending_time)

	''' GRAPHING '''
	x = np.arange(1, 50+1, 1)
	y = []	

	for i in range(50):
		best_fitness = math.inf
		for j in range(NUM_ISLANDS):
			current = BEST_INDICES_GENERATION[i, j].fitness
			if(current < best_fitness):
				best_fitness = current
		y.append(best_fitness)

	plt.plot(x, y)
	plt.xlabel("Num generations")
	plt.ylabel("Best fitness")
	plt.savefig("yeet.png")	
	
	