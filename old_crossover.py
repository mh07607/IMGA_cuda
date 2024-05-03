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
	