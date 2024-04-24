import numpy as np
from random import randint, shuffle, uniform


class GraphColoringGeneticAlgorithm:
    def __init__(self, graph, population_size=60, mutation_chances=[0.65, 0.5, 0.15], num_generations=50):

        self.graph = graph
        self.population_size = population_size
        self.mutation_chances = mutation_chances
        self.num_generations = num_generations
        self.n = len(graph)
        print(self.n)
        self.max_colors = self.get_max_colors()

    def get_max_colors(self): 
        max_num_colors = 0
        for row in self.graph:
            curr_max = sum(row)
            if curr_max > max_num_colors:
                max_num_colors = curr_max
        return max_num_colors

    def create_chromosome(self):
        return np.random.randint(1, self.max_colors + 1, size=self.n) 

    def create_population(self):
        return np.array([self.create_chromosome() for _ in range(self.population_size)])

    def calc_fitness(self, chromosome):
        penalty = 0
        for vertex1 in range(self.n):
            for vertex2 in range(vertex1, self.n):
                if self.graph[vertex1][vertex2] == 1 and chromosome[vertex1] == chromosome[vertex2]:
                    penalty += 1
        return penalty

    def truncation_selection(self, population):
        fitness_result = [self.calc_fitness(chromosome) for chromosome in population]
        sorted_chromosomes = []
        delete = 45
        for _ in range(delete):
            best = fitness_result[0]
            best_idx = 0
            for j in range(len(fitness_result)):
                if fitness_result[j] < best:
                    best = fitness_result[j]
                    best_idx = j
            fitness_result[best_idx] = 0
            sorted_chromosomes.append(population[best_idx])
        return sorted_chromosomes

    def tournament_selection(self, population):
        new_population = []
        for _ in range(2):
            shuffle(population)
            for i in range(0, self.population_size - 1, 2):
                if self.calc_fitness(population[i]) < self.calc_fitness(population[i + 1]):
                    new_population.append(population[i])
                else:
                    new_population.append(population[i + 1])
        return new_population

    def one_point_crossover(self, parent1, parent2):
        split_point = randint(2, self.n - 2)
        child1 = np.concatenate((parent1[:split_point], parent2[split_point:]))
        child2 = np.concatenate((parent2[:split_point], parent1[split_point:]))
        return child1, child2

    def mutation(self, chromosome, chance):
        possible = uniform(0, 1)
        if chance <= possible:
            for vertex1 in range(self.n):
                for vertex2 in range(vertex1, self.n):
                    if self.graph[vertex1][vertex2] == 1 and chromosome[vertex1] == chromosome[vertex2]:
                        chromosome[vertex1] = randint(1, self.max_colors)
        return chromosome

    def evolve(self):
        max_num_colors = self.max_colors
        check_count = 0
        failed_colors = 0

        print(f'Trying to color with {max_num_colors} colors')
        while True:
            population = self.create_population()

            best_fitness = self.calc_fitness(population[0])
            fittest = population[0]

            generation = 0
            num_generations = 50

            if self.n > 5:
                num_generations = self.n * 15

            while best_fitness != 0 and generation != num_generations:
                generation += 1

                population = self.tournament_selection(population)

                if len(population) % 2 != 0:
                    population.pop()

                children_population = []
                shuffle(population)
                for i in range(0, len(population) - 1, 2):
                    child1, child2 = self.one_point_crossover(population[i], population[i + 1])
                    children_population.append(child1)
                    children_population.append(child2)

                for chromosome in children_population:
                    if generation < 200:
                        chromosome = self.mutation(chromosome, self.mutation_chances[0])
                    elif generation < 400:
                        chromosome = self.mutation(chromosome, self.mutation_chances[1])
                    else:
                        chromosome = self.mutation(chromosome, self.mutation_chances[2])

                for i in range(len(population), self.population_size):
                    population.append(self.create_chromosome())

                population = children_population
                best_fitness = self.calc_fitness(population[0])
                fittest = population[0]
                for individual in population:
                    if self.calc_fitness(individual) < best_fitness:
                        best_fitness = self.calc_fitness(individual)
                        fittest = individual

                if best_fitness == 0:
                    break

            if best_fitness == 0:
                print(f'{max_num_colors} colors succeeded! Trying {max_num_colors - 1} colors')
                max_num_colors -= 1
                check_count = 0
            else:
                if check_count != 2 and max_num_colors > 1:
                    failed_colors = max_num_colors

                    if check_count == 0:
                        print(f'{max_num_colors} failed. For safety, checking for improvement with {max_num_colors} colors again')
                    if check_count == 1:
                        print(f'{max_num_colors} failed. For safety, checking for improvement with {max_num_colors - 1} colors')
                        max_num_colors -= 1

                    check_count += 1
                    continue
                if max_num_colors > 1:
                    print(f'Graph is {failed_colors + 1} colorable')
                else:
                    print(f'Graph is {max_num_colors + 1} colorable')
                break

def read_graph_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

        # Initialize variables to store graph size and edges
        n_vertices = 0
        edges = []

        # Iterate through the lines in the file
        for line in lines:
            # Skip comment lines
            if line.startswith('c'):
                continue
            # Parse graph size from the 'p edge' line
            elif line.startswith('p edge'):
                parts = line.split()
                n_vertices = int(parts[2])
            # Parse edge definitions
            elif line.startswith('e'):
                parts = line.split()
                v1 = int(parts[1])
                v2 = int(parts[2])
                edges.append((v1, v2))

        # Create an adjacency matrix representation of the graph
        graph = np.zeros((n_vertices, n_vertices), dtype=int)
        for edge in edges:
            v1, v2 = edge
            graph[v1 - 1][v2 - 1] = 1
            graph[v2 - 1][v1 - 1] = 1

        return graph


if __name__ == '__main__':
    file_path = 'queen11_11.col'  # Replace 'your_graph.col' with the path to your .col file
    graph = read_graph_from_file(file_path)
    genetic_algorithm = GraphColoringGeneticAlgorithm(graph)
    genetic_algorithm.evolve()
    # print(graph)

    