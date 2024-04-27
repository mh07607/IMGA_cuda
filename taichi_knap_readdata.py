import taichi as ti
if __name__ == "__main__":
	ti.init(arch=ti.cpu, default_fp=ti.f64)
from taichi_rng import *
     
knap_sack_items = []

# Open the file in read mode
with open('data/f2_l-d_kp_20_878.txt', 'r') as file:
    # Iterate over each line in the file
    for line in file:
        # Split the line into weight and profit
        profit,weight = map(int, line.split())
        # Append the tuple (weight, profit) to the data list
        knap_sack_items.append((weight, profit))

# print(knap_sack_items)

KNAPSACK_WEIGHT = knap_sack_items[0][0]
TOTAL_ITEMS = knap_sack_items[0][1]

knap_sack_items = knap_sack_items[1:]
KNAPSACK_ITEMS = ti.Vector.field(n=2, dtype=ti.i32, shape=(len(knap_sack_items),))

TYPE_GENOME = ti.types.vector(TOTAL_ITEMS, ti.i32)

for i in range(TOTAL_ITEMS):
    KNAPSACK_ITEMS[i][0] = knap_sack_items[i][0]
    KNAPSACK_ITEMS[i][1] = knap_sack_items[i][1]

@ti.func
def _generate_genome_isl(isl_ind: ti.i32):
    genome = TYPE_GENOME([0 for _ in range(TOTAL_ITEMS)])
    for i in range(TOTAL_ITEMS):
        numb = randint_isl(0,1, isl_ind)
        genome[i] = numb
    return genome

@ti.func
def calculate_value(genome):
    chromo_value = 0
    for i in range(TOTAL_ITEMS):
        if genome[i] == 1:
            chromo_value+= KNAPSACK_ITEMS[i][0]
    return chromo_value

@ti.func
def calculate_weight(genome):
    chromo_weight = 0
    for i in range(TOTAL_ITEMS):
        if genome[i] == 1:
            chromo_weight+= KNAPSACK_ITEMS[i][1]
    return chromo_weight

@ti.func
def calculate_fitness(genome):
    value = calculate_value(genome)
    weight = calculate_weight(genome)
    fitness = ti.f64(ti.math.inf)
    if weight < KNAPSACK_WEIGHT:
        fitness = 1/value
    return fitness
        