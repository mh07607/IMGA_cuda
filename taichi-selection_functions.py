import taichi as ti
from taichi_rng import randint, sample # similar to random.randint and random.sample

@ti.dataclass
class EvolutionaryAlgorithm:
    # # These functions can be set using .methods when used in taichi
    # initial_population_function: FunctionType 
    # parent_selection_function: str,
    # survivor_selection_function: str,
    # cross_over_function: FunctionType, 
    population_size: int = 100,
    mutation_rate: float = 0.5,
    num_offsprings: int = 10
    


@ti.func
def random_selection(self, num_selections: ti.i32): # -> Feild of Indivdual Structs
    survivors = ti.Vector.field(2, dtype=ti.f32, shape=(num_selections,))
    for i in range(num_selections):
        random_int = randint(0, len(self.population)-1)
        survivors[i] = self.population[random_int]
    return survivors



@ti.func
def binary_tournament_selection(self, num_selections: ti.i32):
    result = ti.Vector.field(2, dtype=ti.f64, shape=(num_selections,))
    for i in range(num_selections):
        ind1, ind2 = sample(self.population, 2)
        selected = ind1 if ind1.fitness < ind2.fitness else ind2
        result[i] = selected
    return result
    

if __name__ == "__main__":
    ti.init(arch=ti.gpu, default_fp=ti.f64)
    