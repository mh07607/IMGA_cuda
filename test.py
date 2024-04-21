import random

NUM_CITIES = 194

def TSP_random_length_crossover(parent1, parent2):
	start = random.randint(1, NUM_CITIES-3)
	end = random.randint(start, NUM_CITIES-2)

	offspring1_genome = [-1 for i in range(NUM_CITIES)]
	offspring2_genome = [-1 for i in range(NUM_CITIES)]

	for i in range(start, end+1):
		offspring1_genome[i] = parent1[i]
		offspring2_genome[i] = parent2[i]

	pointer = end + 1
	parent1_pointer = end + 1
	parent2_pointer = end + 1
	# There's a more efficient way to do this I'm sure
	count = end-start+1
	while count < NUM_CITIES:
		gene_found = 0
		for i in range(NUM_CITIES):
			if(offspring1_genome[i] == parent2[parent2_pointer]):
				gene_found = 1
				break

		if not gene_found:
			offspring1_genome[pointer % NUM_CITIES] = parent2[parent2_pointer]
			count += 1
			pointer += 1
		parent2_pointer = (parent2_pointer + 1) % NUM_CITIES

	pointer = end+1
	count = end-start+1
	print(offspring2_genome)
	while count < NUM_CITIES:		
		gene_found = 0
		for i in range(NUM_CITIES):
			if(offspring2_genome[i] == parent1[parent1_pointer]):				
				gene_found = 1
				break

		if not gene_found:			
			offspring2_genome[pointer % NUM_CITIES] = parent1[parent1_pointer]
			count += 1
			pointer += 1
		parent1_pointer = (parent1_pointer + 1) % NUM_CITIES  

	print("crossover range = ", start, end)
	return offspring1_genome, offspring2_genome

individual1 = [(i+1) for i in range(NUM_CITIES)]
individual2 = [(NUM_CITIES) - i for i in range(NUM_CITIES)]

offspring1, offspring2 = TSP_random_length_crossover(individual1, individual2)
print(offspring1)
print(offspring2)
print("original")
#offspring3, offspring4 = original_TSP_random_length_crossover(individual1, individual2)
#print(offspring3)
#print(offspring4)