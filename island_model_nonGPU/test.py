from random import randint
import random
import numpy as np
from itertools import tee

# Change n to change graph size
n = 121
popSize = 60

#! ============== GRAPH AND POPULATION CREATION FUNCTIONS
def createGraph():
    graph = np.random.randint(0, 2, size=(n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                graph[i][j] = 0
                continue
            graph[i][j] = graph[j][i]

    return graph

def read_graph_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        n_vertices = 0
        edges = []
        for line in lines:
            if line.startswith('c'):
                continue
            elif line.startswith('p edge'):
                parts = line.split()
                n_vertices = int(parts[2])
            elif line.startswith('e'):
                parts = line.split()
                v1 = int(parts[1])
                v2 = int(parts[2])
                edges.append((v1, v2))
        graph = np.zeros((n_vertices, n_vertices), dtype=int)
        for edge in edges:
            v1, v2 = edge
            graph[v1 - 1][v2 - 1] = 1
            graph[v2 - 1][v1 - 1] = 1

        return graph

def printGraph(graph):
    print([row for row in graph])

def getMaxColors(graph):
    maxNumColors = 0
    for row in graph:
        currMax = sum(row)
        if currMax > maxNumColors:
            maxNumColors = currMax
    return maxNumColors

def createChromosome(maxNumColors):
    # Each solution is represented in a 1D array where the index of each item in that array maps to the index of a vertex in the graph
    return np.random.randint(1, maxNumColors + 1, size=(n))

def createPopulation(maxNumColors):
    return np.array([createChromosome(maxNumColors) for i in range(popSize)])


#! ============== FITNESS FUNCTION
def calcFitness(graph, chromosome):
    penalty = 0
    for vertex1 in range(n):
        for vertex2 in range(vertex1, n):
            if graph[vertex1][vertex2] == 1 and chromosome[vertex1] == chromosome[vertex2]:
                penalty += 1
    return penalty


#! ============== SELECTION
def truncationSelection(population):
    fitnessResult = [calcFitness(graph, chromosome) for chromosome in population]
    sortedChromosomes = []
    delete = 45
    for i in range(delete):
        best = fitnessResult[0]
        bestIdx = 0
        for j in range(len(fitnessResult)):
            if fitnessResult[j] < best:
                best = fitnessResult[j]
                bestIdx = j
        fitnessResult[bestIdx] = 0
        sortedChromosomes.append(population[bestIdx])
    return sortedChromosomes

def tournamentSelection(population):
    newPopulation = []
    for _ in range(2):
        random.shuffle(population)
        for i in range(0, popSize - 1, 2):
            if calcFitness(graph, population[i]) < calcFitness(graph, population[i + 1]):
                newPopulation.append(population[i])
            else:
                newPopulation.append(population[i + 1])
    return newPopulation


#! ============== CROSSOVER
def onePointCrossover(parent1, parent2):
    splitPoint = randint(2, n - 2)
    child1 = np.concatenate((parent1[:splitPoint], parent2[splitPoint:]))
    child2 = np.concatenate((parent2[:splitPoint], parent1[splitPoint:]))
    return child1, child2

def twoPointCrossover(parent1, parent2):
    firstPoint = randint(1, n - 3)
    secondPoint = randint(firstPoint + 1, n - 1)
    child1 = np.concatenate((parent1[:firstPoint], parent2[firstPoint:secondPoint], parent1[secondPoint:]))
    child2 = np.concatenate((parent2[:firstPoint], parent1[firstPoint:secondPoint], parent2[secondPoint:]))
    return child1, child2


#! ============== MUTATION
def mutation(chromosome, chance):    
    # Find invalid vertex to mutate
    possible = random.uniform(0, 1)
    if chance <= possible:
        for vertex1 in range(n):
            for vertex2 in range(vertex1, n):
                if graph[vertex1][vertex2] == 1 and chromosome[vertex1] == chromosome[vertex2]:
                    chromosome[vertex1] = randint(1, maxNumColors)
    return chromosome


#! --------- TEST METHODS
def generateTestGraph():
    global graph
    result = {}

    for i in range(n):
        result[i] = []
        for j in range(n):
            if graph[i][j] == 1:
                result[i].append(j)

    return result

def maxColoring(colorDict):
    maxColors = max(colorDict.values())
    return maxColors

# This method was obtained from https://www.codespeedy.com/graph-coloring-using-greedy-method-in-python/
def testColoring():
  graph = generateTestGraph()
  vertices = sorted((list(graph.keys())))
  colour_graph = {}

  for vertex in vertices:
    unused_colours = len(vertices) * [True]

    for neighbor in graph[vertex]:
      if neighbor in colour_graph:
        colour = colour_graph[neighbor]
        unused_colours[colour] = False
    for colour, unused in enumerate(unused_colours):
        if unused:
            colour_graph[vertex] = colour
            break

  return maxColoring(colour_graph)
#! ------------- END TEST METHODS 



if __name__ == '__main__':
    #graph = createGraph()
    file_path = '/Users/asadullahchaudhry/Github HU/IMGA_cuda/island_model_nonGPU/queen11_11.col'  
    graph = read_graph_from_file(file_path)
    

    # Some base cases that do not need serious computing 
    if n == 1:
        print('Graph is 1 colorable')
    elif n == 2:
        if 1 in graph[0]:
            print('Graph is 2 colorable')
        else:
            print('Graph is 1 colorable')
    else:
        maxNumColors = getMaxColors(graph)

        checkCount = 0
        failedColors = 0

        print(f'Trying to color with {maxNumColors} colors')
        while True:
            population = createPopulation(maxNumColors)

            bestFitness = calcFitness(graph, population[0])
            fittest = population[0]

            # Generation Control
            generation = 0
            numGenerations = 50

            if n > 5:
                numGenerations = n * 15

            while bestFitness != 0 and generation != numGenerations:
                print(bestFitness)
                generation += 1

                #! SELECTION
                population = tournamentSelection(population)
                # population = truncationSelection(population)

                # Make sure population is even 
                if len(population) % 2 != 0:
                    population.pop()

                #! CROSSOVER
                childrenPopulation = []
                random.shuffle(population)
                for i in range(0, len(population) - 1, 2):
                    child1, child2 = onePointCrossover(population[i], population[i + 1])
                    childrenPopulation.append(child1)
                    childrenPopulation.append(child2)

                #! MUTATION
                for chromosome in childrenPopulation:
                    if generation < 200:
                        chromosome = mutation(chromosome, 0.65)
                    elif generation < 400:
                        chromosome = mutation(chromosome, 0.5)
                    else:
                        chromosome = mutation(chromosome, 0.15)

                #! Fill up the rest of the population with random values
                for i in range(len(population), popSize):
                    population.append(createChromosome(maxNumColors))

                #! FITNESS
                population = childrenPopulation
                bestFitness = calcFitness(graph, population[0])
                fittest = population[0]
                for individual in population:
                    if(calcFitness(graph, individual) < bestFitness):
                        bestFitness = calcFitness(graph, individual)
                        fittest = individual

                if bestFitness == 0:
                    break

                # if generation % 10 == 0:
                #     print(f'generationeration: {generation}, Best Fitness: {bestFitness}, Individual: {fittest}')

            # print(f'Using {maxNumolorsx} colors')
            if bestFitness == 0:
                print(f'{maxNumColors} colors succeeded! Trying {maxNumColors - 1} colors')
                maxNumColors -= 1
                checkCount = 0
            else:
                if checkCount != 2 and maxNumColors > 1:
                    failedColors = maxNumColors
                    
                    if checkCount == 0:
                        print(f'{maxNumColors} failed. For safety, checking for improvement with {maxNumColors} colors again')
                    if checkCount == 1:
                        print(f'{maxNumColors} failed. For safety, checking for improvement with {maxNumColors - 1} colors')
                        maxNumColors -= 1

                    checkCount += 1
                    continue
                if maxNumColors > 1:
                    print(f'Graph is {failedColors + 1} colorable')
                else:
                    print(f'Graph is {maxNumColors + 1} colorable')
                print('Test Solution:', testColoring())
                break