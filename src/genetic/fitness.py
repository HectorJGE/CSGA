import numpy as np

# Número de usuarios simulados
NUM_USERS = 50
# Umbral para considerar que le "gustó" una película
LIKE_THRESHOLD = 3.5
# Número de recomendaciones a generar
TOP_N = 10

def evaluate(individual, ratings_df):
    movie_ids = individual.movie_ids
    all_user_ids = ratings_df['userId'].unique()
    
    # Elegimos usuarios que hayan calificado al menos una película del conjunto
    good_users = []
    for uid in all_user_ids:
        user_ratings = ratings_df[ratings_df['userId'] == uid]
        if user_ratings['movieId'].isin(movie_ids).sum() >= 1:
            good_users.append(uid)
        if len(good_users) >= NUM_USERS:
            break
    
    if len(good_users) == 0:
        individual.fitness = 0
        return 0

    fitness_scores = []

    for uid in good_users:
        user_data = ratings_df[ratings_df['userId'] == uid]

        # Simulamos que sólo "conocemos" sus ratings sobre las películas del conjunto
        known_ratings = user_data[user_data['movieId'].isin(movie_ids)]
        liked_movies = known_ratings[known_ratings['rating'] >= LIKE_THRESHOLD]['movieId'].tolist()

        # print(f"\nEvaluando usuario {uid}")
        # print(f"- Películas del conjunto que valoró: {known_ratings['movieId'].tolist()}")
        # print(f"- Le gustaron (rating ≥ {LIKE_THRESHOLD}): {liked_movies}")
        
        if not liked_movies:
            continue

        # Encontrar usuarios similares (los que también calificaron esas películas)
        similar_users = ratings_df[
            (ratings_df['movieId'].isin(liked_movies)) &
            (ratings_df['userId'] != uid)
        ]

        # print(f"👥 Usuarios similares encontrados: {len(similar_users)}")
        # print(similar_users[['userId', 'movieId', 'rating']].head(5))

        # if similar_users.empty:
        #     print("No hay usuarios similares, no se puede recomendar.")
        #     continue

        known_ids = set(movie_ids)

        # 🧠 Nueva lógica: buscar lo que calificaron bien los usuarios similares
        similar_ids = similar_users['userId'].unique()
        similars_ratings = ratings_df[ratings_df['userId'].isin(similar_ids)]

        reco_candidates = similars_ratings[
            (~similars_ratings['movieId'].isin(known_ids)) &
            (similars_ratings['rating'] >= LIKE_THRESHOLD)
        ]

        recommended = (
            reco_candidates
            .groupby('movieId')['rating']
            .mean()
            .sort_values(ascending=False)
            .head(TOP_N)
            .index.tolist()
        )

        # print(f" Películas recomendadas: {recommended}")

        # Películas que al usuario realmente le gustaron (fuera del conjunto conocido)
        true_likes = user_data[
            (~user_data['movieId'].isin(movie_ids)) &
            (user_data['rating'] >= LIKE_THRESHOLD)
        ]['movieId'].tolist()

        # Precisión = cuántas recomendaciones realmente le gustaron
        if true_likes:
            hits = len(set(recommended) & set(true_likes))
            precision = hits / TOP_N
            fitness_scores.append(precision)

    # Fitness = promedio de precisión
    if fitness_scores:
        individual.fitness = np.mean(fitness_scores)
    else:
        individual.fitness = 0

    return individual.fitness