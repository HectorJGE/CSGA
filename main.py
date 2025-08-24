import random
import os
import numpy as np
import pandas as pd
from src.utils.data_loader import load_movies, build_item_pool
from src.genetic.population import Population
from src.genetic.fitness import init_fitness_cache, init_content_cache, evaluate

POP_SIZE = 10
ITEMS_PER_SET = 5
N_GENERATIONS = 5
SEED = 42

def set_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

set_seed(SEED)

rng = np.random.default_rng(SEED)


def rand_choice(pool, k):
    idx = rng.choice(len(pool), size=k, replace=False)
    return [pool[i] for i in idx]

def mostrar_peliculas(individual, df_movies):
    print("\nPelículas del mejor individuo:")
    for movie_id in individual.movie_ids:
        row = df_movies[df_movies['movieId'] == movie_id]
        if row.empty:
            continue
        title = row.iloc[0]['title']
        genres = row.iloc[0]['genres']
        if isinstance(genres, str):
            genres = genres.split('|') if genres else []
        print(f"- {title}  ({', '.join(genres)})")

# Cargar datos
df_movies = load_movies("data/ml-latest-small/movies.csv")
df_ratings = pd.read_csv("data/ml-latest-small/ratings.csv")

# Pool de ítems
item_pool = build_item_pool(df_movies)
assert ITEMS_PER_SET <= len(item_pool), "ITEMS_PER_SET > tamaño del pool."

# Población inicial
population = Population(POP_SIZE, item_pool, ITEMS_PER_SET)
population.set_ratings(df_ratings)

# Evaluar población inicial (si evolve no evalúa)

init_fitness_cache(df_ratings, m_wr=50, like_threshold=3.5)
init_content_cache(df_movies, df_ratings, tau_year=6.0)

for ind in population.individuals:
    evaluate(ind, df_ratings)

# Evolución
for gen in range(N_GENERATIONS):
    print(f"\nGeneración {gen+1}")
    population.evolve()
    # Si evolve NO evalúa internamente, reevalúa acá:
    if any(getattr(ind, 'fitness', None) is None for ind in population.individuals):
        for ind in population.individuals:
            evaluate(ind, df_ratings)
    best = max(population.individuals, key=lambda ind: ind.fitness)
    print("Mejor fitness:", round(best.fitness, 4))
    components = getattr(best, "fitness_components", None)
    if components:
        print("Componentes ->",
            "precision:", round(components["precision"], 4),
            "| diversity:", round(components["diversity"], 4),
            "| entropy:", round(components["entropy"], 4),
            "| novelty:", round(components["novelty"], 4))
    mostrar_peliculas(best, df_movies)
