import taichi as ti
from taichi_rng import *

file_path = "data/queen11_11.col"
NUM_VERTICES = 0
edges = []
with open(file_path, 'r') as file:
        lines = file.readlines()        
        for line in lines:
            if line.startswith('c'):
                continue
            elif line.startswith('p edge'):
                parts = line.split()
                NUM_VERTICES = int(parts[2])
            elif line.startswith('e'):
                parts = line.split()
                v1 = int(parts[1])
                v2 = int(parts[2])
                edges.append((v1, v2))

# graph = np.zeros((NUM_VERTICES, NUM_VERTICES), dtype=int)
GRAPH = ti.field(dtype=ti.f64, shape=(NUM_VERTICES, NUM_VERTICES))
for edge in edges:
    v1, v2 = edge
    GRAPH[v1 - 1, v2 - 1] = 1
    GRAPH[v2 - 1, v1 - 1] = 1

TYPE_GENOME = ti.types.vector(NUM_VERTICES, ti.i32)

MAX_COLORS = 0

# for row in GRAPH:
#     curr_max = sum(row)
#     if curr_max > MAX_COLORS:
#         MAX_COLORS = curr_max    

for i in range(NUM_VERTICES):
    curr_max = 0
    for j in range(NUM_VERTICES):
        curr_max += GRAPH[i, j]
    if(curr_max > MAX_COLORS):
        MAX_COLORS = curr_max

@ti.func
def _generate_genome_isl(isl_ind: ti.i32):
    genome = TYPE_GENOME([0 for _ in range(NUM_VERTICES)])
    for i in range(NUM_VERTICES):
        numb = randint_isl(1, MAX_COLORS, isl_ind)
        genome[i] = numb
    return genome

@ti.func
def calculate_fitness(genome):
    penalty = 0
    for vertex1 in range(NUM_VERTICES):
        for vertex2 in range(vertex1, NUM_VERTICES):
            if GRAPH[vertex1, vertex2] == 1 and genome[vertex1] == genome[vertex2]:
                penalty += 1
    return penalty
