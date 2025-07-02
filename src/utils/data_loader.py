import pandas as pd

def load_movies(path='data/movies.csv'):
    df = pd.read_csv(path)
    
    # Separar géneros
    df['genres'] = df['genres'].apply(lambda g: g.split('|') if isinstance(g, str) else [])
    
    return df

def build_item_pool(df_movies):
    return [
        {
            "id": row.movieId,
            "title": row.title,
            "genres": row.genres
        }
        for _, row in df_movies.iterrows()
    ]