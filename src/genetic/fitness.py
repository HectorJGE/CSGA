from collections import defaultdict
import numpy as np
import pandas as pd

# Caches de módulo (se rellenan con init_fitness_cache)
_FCACHE = {
    "user_likes": None,   # dict: userId -> set(movieId con "like")
    "movie_users": None,  # dict: movieId -> set(userId que dieron "like")
    "wr": None,           # dict: movieId -> weighted rating (IMDB-like)
    "pop": None,          # dict: movieId -> log1p(#ratings)
    "user_mean": None,    # dict: userId -> promedio de rating (por si luego usás coseno ajustado)
}

def _compute_wr_and_pop(ratings_df: pd.DataFrame, m: int = 50) -> tuple[dict, dict]:
    grp = ratings_df.groupby("movieId")["rating"]
    v = grp.count()        # #ratings por película
    R = grp.mean()         # rating promedio por película
    C = ratings_df["rating"].mean()  # promedio global
    wr = (v / (v + m)) * R + (m / (v + m)) * C
    pop = np.log1p(v)      # popularidad log-transformada
    return wr.to_dict(), pop.to_dict()

def _build_binary_structures(ratings_df: pd.DataFrame, like_threshold: float = 3.5):
    df = ratings_df.copy()
    df["like"] = (df["rating"] >= like_threshold).astype(int)
    df_like = df[df["like"] == 1][["userId", "movieId"]]

    user_likes = df_like.groupby("userId")["movieId"].apply(set).to_dict()
    movie_users = df_like.groupby("movieId")["userId"].apply(set).to_dict()
    user_mean = ratings_df.groupby("userId")["rating"].mean().to_dict()
    return user_likes, movie_users, user_mean

def init_fitness_cache(ratings_df: pd.DataFrame, m_wr: int = 50, like_threshold: float = 3.5):
    user_likes, movie_users, user_mean = _build_binary_structures(ratings_df, like_threshold)
    wr, pop = _compute_wr_and_pop(ratings_df, m=m_wr)
    _FCACHE["user_likes"] = user_likes
    _FCACHE["movie_users"] = movie_users
    _FCACHE["wr"] = wr
    _FCACHE["pop"] = pop
    _FCACHE["user_mean"] = user_mean

def _extract_genres_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        return x.split("|") if x else []
    return []

def _idf_per_genre(df_movies):
    # df_movies['genres'] puede ser string "A|B|C" o lista; contamos df por película
    all_genres = []
    N = len(df_movies)
    for g in df_movies['genres']:
        gl = _extract_genres_list(g)
        all_genres.extend(set(gl))  # cada película aporta sus géneros una vez
    # df por género
    from collections import Counter
    df_counter = Counter(all_genres)
    # idf suavizado: log((N+1)/(df+1)) + 1
    idf = {g: log((N + 1) / (df_counter[g] + 1)) + 1.0 for g in df_counter}
    return idf

def _build_content_structures(df_movies: pd.DataFrame, ratings_df: pd.DataFrame, tau_year: float = 6.0):
    # Mapeos básicos
    movie_year = {}
    movie_gen_vec = {}  # vector de géneros ponderado por IDF y normalizado L2
    idf = _idf_per_genre(df_movies)

    for _, row in df_movies.iterrows():
        mid = int(row["movieId"])
        # Año: intenta extraer del título (ej: "Toy Story (1995)")
        title = row.get("title", "")
        year = None
        if isinstance(title, str) and "(" in title and ")" in title:
            try:
                year = int(title.strip()[-5:-1])
            except Exception:
                year = None
        movie_year[mid] = year

        # Vector de géneros
        gl = _extract_genres_list(row.get("genres", ""))
        if not gl:
            movie_gen_vec[mid] = {}
            continue
        # Pesar por IDF
        vec = {g: idf.get(g, 1.0) for g in gl}
        # Normalizar L2
        norm = sqrt(sum(v*v for v in vec.values()))
        if norm > 0:
            vec = {k: v / norm for k, v in vec.items()}
        movie_gen_vec[mid] = vec

    # Recency normalizada por último timestamp
    if "timestamp" in ratings_df.columns:
        last_ts = ratings_df.groupby("movieId")["timestamp"].max()
        if len(last_ts) > 0:
            min_t, max_t = float(last_ts.min()), float(last_ts.max())
            if max_t > min_t:
                rec_norm = ((last_ts - min_t) / (max_t - min_t)).to_dict()
            else:
                rec_norm = {int(k): 0.0 for k in last_ts.index}
        else:
            rec_norm = {}
    else:
        rec_norm = {}

    return {
        "movie_year": movie_year,
        "movie_gen_vec": movie_gen_vec,
        "recency_norm": rec_norm,
        "tau_year": tau_year,
    }

def init_content_cache(df_movies: pd.DataFrame, ratings_df: pd.DataFrame, tau_year: float = 6.0):
    C = _build_content_structures(df_movies, ratings_df, tau_year)
    _FCACHE["movie_year"] = C["movie_year"]
    _FCACHE["movie_gen_vec"] = C["movie_gen_vec"]
    _FCACHE["recency_norm"] = C["recency_norm"]
    _FCACHE["tau_year"] = C["tau_year"]
    # Normalizamos popularidad a [0,1] para híbrido/novelty
    pop = _FCACHE.get("pop", {})
    if pop:
        pv = list(pop.values())
        pmin, pmax = min(pv), max(pv)
        if pmax > pmin:
            _FCACHE["pop_norm"] = {k: (v - pmin) / (pmax - pmin) for k, v in pop.items()}
        else:
            _FCACHE["pop_norm"] = {k: 0.0 for k in pop.keys()}
    else:
        _FCACHE["pop_norm"] = {}

def _cosine_gen(mid_i: int, mid_j: int) -> float:
    gi = _FCACHE["movie_gen_vec"].get(mid_i, {})
    gj = _FCACHE["movie_gen_vec"].get(mid_j, {})
    if not gi or not gj:
        return 0.0
    # producto punto sobre claves comunes (ya normalizados L2)
    common = set(gi.keys()) & set(gj.keys())
    if not common:
        return 0.0
    return sum(gi[g]*gj[g] for g in common)

def _sim_year(mid_i: int, mid_j: int) -> float:
    yi = _FCACHE["movie_year"].get(mid_i, None)
    yj = _FCACHE["movie_year"].get(mid_j, None)
    if yi is None or yj is None:
        return 0.0
    tau = _FCACHE.get("tau_year", 6.0)
    return exp(-abs(yi - yj) / max(tau, 1e-6))

def _s_content(mid_i: int, mid_j: int, w_gen: float = 0.7, w_year: float = 0.3) -> float:
    # ambas similares en [0,1], pesos a gusto
    return w_gen * _cosine_gen(mid_i, mid_j) + w_year * _sim_year(mid_i, mid_j)

def _fallback_content_from_anchor(anchor_likes: set[int], exclude_ids: set[int], top_n: int = 10,
                                  w_gen: float = 0.7, w_year: float = 0.3,
                                  gamma_pop: float = 0.0, delta_rec: float = 0.0):
    """
    Genera candidatos por contenido promediando similitud con las anclas.
    Re-rankea con pop_norm y recency_norm si gamma/delta > 0.
    """
    # Candidatos: todas las pelis que comparten al menos un género con alguna ancla (recorte práctico)
    movie_gen_vec = _FCACHE["movie_gen_vec"]
    pop_norm = _FCACHE.get("pop_norm", {})
    rec_norm = _FCACHE.get("recency_norm", {})

    candidate_set = set()
    anchor_genes = set()
    for a in anchor_likes:
        anchor_genes |= set(movie_gen_vec.get(a, {}).keys())
    if not anchor_genes:
        return []

    for mid, vec in movie_gen_vec.items():
        if mid in exclude_ids:
            continue
        if set(vec.keys()) & anchor_genes:
            candidate_set.add(mid)

    scored = []
    for c in candidate_set:
        # promedio de s_content con todas las anclas (también podés usar max)
        sims = []
        for a in anchor_likes:
            sims.append(_s_content(a, c, w_gen, w_year))
        scont = float(np.mean(sims)) if sims else 0.0
        # híbrido simple con pop/recency si se pide
        score = scont + gamma_pop * pop_norm.get(c, 0.0) + delta_rec * rec_norm.get(c, 0.0)
        scored.append((c, score, scont, pop_norm.get(c, 0.0), rec_norm.get(c, 0.0)))

    scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return scored[:top_n]

# Número de usuarios simulados
NUM_USERS = 50
# Umbral para considerar que le "gustó" una película
LIKE_THRESHOLD = 3.5
# Número de recomendaciones a generar
TOP_N = 10

def _sample_good_users(individual, min_likes_in_set: int = 2, max_users: int = 200, seed: int = 42):
    user_likes = _FCACHE["user_likes"]
    ids_set = set(individual.movie_ids)

    # Elegibles: usuarios que tienen al menos 'min_likes_in_set' likes dentro del conjunto candidato
    eligible = [u for u, likes in user_likes.items() if len(likes & ids_set) >= min_likes_in_set]

    # Muestreo reproducible
    rng = np.random.default_rng(seed)
    if len(eligible) > max_users:
        eligible = list(rng.choice(eligible, size=max_users, replace=False))
    return eligible

def _rank_candidates_for_user(
    user_id: int,
    anchor_likes: set[int],
    exclude_ids: set[int],
    lambda_shrink: int = 25,
    min_support: int = 5,
    top_n: int = 10,
):
    user_likes = _FCACHE["user_likes"]
    movie_users = _FCACHE["movie_users"]
    wr = _FCACHE["wr"]
    pop = _FCACHE["pop"]

    # Usuarios similares: unión de usuarios que likearon cualquiera de las anclas
    sim_users = set()
    for m in anchor_likes:
        sim_users |= movie_users.get(m, set())
    sim_users.discard(user_id)
    n = len(sim_users)
    if n == 0:
        return []

    # Conteo de likes por candidato entre los similares
    from collections import defaultdict
    counts = defaultdict(int)
    for su in sim_users:
        for mov in user_likes.get(su, set()):
            if mov not in exclude_ids:
                counts[mov] += 1

    scored = []
    for mov, c in counts.items():
        if c < min_support:
            continue
        # Probabilidad empírica con shrinkage por soporte
        p = c / n
        s = (n / (n + lambda_shrink)) * p
        scored.append((mov, s, wr.get(mov, 0.0), pop.get(mov, 0.0)))

    # Orden: score desc, WR desc, pop desc
    scored.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
    return scored[:top_n]

def _precision_at_k(recommended_ids: list[int], user_liked: set[int], exclude_ids: set[int]) -> float:
    if not recommended_ids:
        return 0.0
    hits = sum(1 for mid in recommended_ids if mid in user_liked and mid not in exclude_ids)
    denom = len(recommended_ids)  # usa el K real (<= top_n)
    return hits / denom

def _diversity_of_set(movie_ids: list[int]) -> float:
    # Diversidad por contenido: 1 - s_content promedio de pares
    if len(movie_ids) < 2:
        return 0.0
    pairs = 0
    acc = 0.0
    for i in range(len(movie_ids)):
        for j in range(i+1, len(movie_ids)):
            s = _s_content(movie_ids[i], movie_ids[j])
            acc += (1.0 - s)
            pairs += 1
    return acc / pairs if pairs > 0 else 0.0

def _entropy_of_genres(movie_ids: list[int]) -> float:
    # Entropía Shannon sobre distribución de géneros (ponderada por IDF), normalizada por log(|G|)
    movie_gen_vec = _FCACHE["movie_gen_vec"]
    # suma de pesos por género en el set (ya están normalizados por película)
    from collections import Counter
    agg = Counter()
    for mid in movie_ids:
        for g, w in movie_gen_vec.get(mid, {}).items():
            agg[g] += w
    total = sum(agg.values())
    if total <= 0:
        return 0.0
    probs = [v/total for v in agg.values()]
    H = -sum(p * log(p + 1e-12) for p in probs)
    Hmax = log(len(probs)) if len(probs) > 0 else 1.0
    return float(H / Hmax) if Hmax > 0 else 0.0

def _novelty_of_set(movie_ids: list[int]) -> float:
    pop_norm = _FCACHE.get("pop_norm", {})
    vals = [1.0 - pop_norm.get(mid, 0.0) for mid in movie_ids]
    return float(np.mean(vals)) if vals else 0.0

def evaluate(individual, ratings_df=None,
             min_likes_in_set=2, max_users=200, seed=42,
             lambda_shrink=25, min_support=5, top_n=10,
             # pesos del fitness total
             w_precision=0.6, w_div=0.2, w_entropy=0.1, w_novel=0.1,
             # híbrido / fallback de contenido
             w_gen=0.7, w_year=0.3, gamma_pop=0.0, delta_rec=0.0,
             ):
    assert _FCACHE["user_likes"] is not None, "Llamá init_fitness_cache(...) antes."
    ids_set = set(individual.movie_ids)
    good_users = _sample_good_users(individual, min_likes_in_set, max_users, seed)

    precisions = []
    for uid in good_users:
        anchor = _FCACHE["user_likes"].get(uid, set()) & ids_set
        exclude = set(individual.movie_ids)

        ranked = _rank_candidates_for_user(
            user_id=uid,
            anchor_likes=anchor,
            exclude_ids=exclude,
            lambda_shrink=lambda_shrink,
            min_support=min_support,
            top_n=top_n,
        )
        recs = [mid for (mid, s, wr, pop) in ranked]

        # Fallback de contenido si no hay colaborativo confiable
        if not recs:
            fallback = _fallback_content_from_anchor(
                anchor_likes=anchor,
                exclude_ids=exclude,
                top_n=top_n,
                w_gen=w_gen, w_year=w_year,
                gamma_pop=gamma_pop, delta_rec=delta_rec
            )
            recs = [mid for (mid, score, scont, p, r) in fallback]

        prec = _precision_at_k(recs, _FCACHE["user_likes"].get(uid, set()), exclude)
        precisions.append(prec)

    precision_mean = float(np.mean(precisions)) if precisions else 0.0

    # Métricas intra-set
    div = _diversity_of_set(list(ids_set))
    ent = _entropy_of_genres(list(ids_set))
    nov = _novelty_of_set(list(ids_set))

    fitness = (
        w_precision * precision_mean +
        w_div * div +
        w_entropy * ent +
        w_novel * nov
    )
    individual.fitness = float(fitness)
    # opcional: guardar componentes para logging/diagnóstico
    individual.fitness_components = {
        "precision": precision_mean,
        "diversity": div,
        "entropy": ent,
        "novelty": nov
    }
    return individual.fitness
