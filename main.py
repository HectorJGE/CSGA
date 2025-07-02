from src.genetic.population import Population
from src.genetic.fitness import evaluate
from src.utils.data_loader import load_movies, build_item_pool

# Configuración
POP_SIZE = 50
N_GENERATIONS = 100
ITEMS_PER_SET = 5

# Cargar ítems

df_movies = load_movies("data/ml-latest-small/movies.csv")
# Ver columnas y primeras filas
print(df_movies.head())

# Verificar nulos
print(df_movies.isnull().sum())

item_pool = build_item_pool(df_movies)
# print(f"Total de ítems disponibles: {len(item_pool)}")

# Crear población inicial
population = Population(POP_SIZE, item_pool, ITEMS_PER_SET)

# Evaluar y evolucionar
# for generation in range(N_GENERATIONS):
#     for ind in population.individuals:
#         evaluate(ind, item_pool)
#     population = population.evolve()

# Resultado final
# best = max(population.individuals, key=lambda ind: ind.fitness)
# print("Mejor set:", best.items)