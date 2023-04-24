import numpy as np
import pandas as pd


def load_users(path_to_file: str):
    df_users = pd.read_csv(path_to_file, dtype={"age": "category"})
    df_users["sex"] = df_users["sex"].astype(pd.SparseDtype(np.float32, np.nan))
    df_users = df_users.set_index("user_id", verify_integrity=True)

    return df_users


def sample_true_rec_data():
    df_true = pd.DataFrame({
        'user_id': ['Аня',                'Боря',               'Вася',         'Вася'],
        'item_id': ['Мастер и Маргарита', '451° по Фаренгейту', 'Зеленая миля', 'Рита Хейуорт и спасение из Шоушенка'],
    }
    )

    df_true.set_index(["user_id", "item_id"], inplace=True)

    df_recs = pd.DataFrame({
        'user_id': [
            'Аня', 'Аня', 'Аня',
            'Боря', 'Боря', 'Боря',
            'Вася', 'Вася', 'Вася',
        ],
        'item_id': [
            'Отверженные', 'Двенадцать стульев', 'Герои нашего времени',
            '451° по Фаренгейту', '1984', 'О дивный новый мир',
            'Десять негритят', 'Искра жизни', 'Зеленая миля',
        ],
        'rank': [
            1, 2, 3,
            1, 2, 3,
            1, 2, 3,
        ]
    }
    )

    df_recs.set_index(["user_id", "item_id"], inplace=True)

    return df_true, df_recs


def load_interactions(path_to_file: str):
    interactions = pd.read_csv(path_to_file, parse_dates=["start_date"])
    duplicates = interactions.duplicated(subset=['user_id', 'item_id'], keep=False)
    df_duplicates = interactions[duplicates].sort_values(by=['user_id', 'start_date'])
    interactions = interactions[~duplicates]

    df_duplicates = df_duplicates.groupby(['user_id', 'item_id']).agg({
        'progress': 'max',
        'rating': 'max',
        'start_date': 'min'
    })

    interactions = pd.concat((interactions, df_duplicates.reset_index()), ignore_index=True)
    interactions['progress'] = interactions['progress'].astype(np.int8)
    interactions['rating'] = interactions['rating'].astype(pd.SparseDtype(np.float32, np.nan))

    interactions = interactions.set_index(["user_id", "item_id"], verify_integrity=True)

    return interactions


def load_items(path_to_file: str):
    items = pd.read_csv(path_to_file, dtype={
                        "genres": "category", "authors": "category", "year": "category"})

    items = items.set_index("id", verify_integrity=True)
    items.index = items.index.set_names("item_id")
    return items
