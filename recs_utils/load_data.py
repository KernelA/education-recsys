import numpy as np
import pandas as pd
import polars as pl

from .metrics import USER_ID_COL, ITEM_ID_COL


def load_users(path_to_file: str):
    df_users = pl.read_csv(path_to_file, dtypes={
                           "age": pl.Categorical, "user_id": pl.UInt32, "sex": pl.Float32})
    return df_users


def sample_true_rec_data():
    df_true = pl.DataFrame({
        'user_id': ['Аня',                'Боря',               'Вася',         'Вася'],
        'item_id': ['Мастер и Маргарита', '451° по Фаренгейту', 'Зеленая миля', 'Рита Хейуорт и спасение из Шоушенка'],
    }
    )

    df_recs = pl.DataFrame({
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
    return df_true, df_recs


def load_interactions(path_to_file: str):
    interactions = pl.read_csv(path_to_file,
                               dtypes={
                                   "user_id": pl.UInt32,
                                   "item_id": pl.UInt32,
                                   "start_date": pl.Date,
                                   "rating": pl.Float32,
                                   "progress": pl.UInt8
                               }).with_columns(pl.col("rating").fill_null(float("nan")))

    uniq_items = interactions.select(
        pl.col("user_id"),
        pl.col("item_id")
    )

    not_duplicated_interactions = interactions.filter(uniq_items.is_unique())
    duplicated_interactions = interactions.filter(uniq_items.is_duplicated()).groupby(["user_id", "item_id"]).agg(
        [
            pl.max("progress"),
            pl.max("rating"),
            pl.min("start_date")
        ]
    )

    return not_duplicated_interactions.vstack(duplicated_interactions)


def load_items(path_to_file: str):
    return pl.read_csv(
        path_to_file,
        dtypes={
            "id": pl.UInt32,
            "genres": pl.Categorical,
            "authors": pl.Categorical,
            "year": pl.Categorical
        },
        new_columns=["item_id"]
    )
