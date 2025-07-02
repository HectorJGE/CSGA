import pandas as pd
from src.utils.data_loader import load_movies, build_item_pool
from src.genetic.population import Population
from src.genetic.fitness import evaluate

POP_SIZE = 10
ITEMS_PER_SET = 5
N_GENERATIONS = 5

def mostrar_peliculas(individual, df_movies):
    print("\nPelículas del mejor individuo:")
    for movie_id in individual.movie_ids:
        row = df_movies[df_movies['movieId'] == movie_id]
        if not row.empty:
            title = row.iloc[0]['title']
            genres = ", ".join(row.iloc[0]['genres'])
            print(f"- {title}  ({genres})")

# Cargar datos
df_movies = load_movies("data/ml-latest-small/movies.csv")
df_ratings = pd.read_csv("data/ml-latest-small/ratings.csv")
item_pool = build_item_pool(df_movies)

population = Population(POP_SIZE, item_pool, ITEMS_PER_SET)
population.set_ratings(df_ratings)

# Evaluar población inicial
for ind in population.individuals:
    evaluate(ind, df_ratings)

# Evolución
for gen in range(N_GENERATIONS):
    print(f"\nGeneración {gen+1}")
    population.evolve()
    best = max(population.individuals, key=lambda ind: ind.fitness)
    print("Mejor fitness:", best.fitness)
    mostrar_peliculas(best, df_movies)