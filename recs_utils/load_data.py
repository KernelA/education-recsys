from typing import List, Optional

import polars as pl

from .metrics import USER_ID_COL, ITEM_ID_COL


class MovieLens100K:
    """https://www.kaggle.com/datasets/prajitdatta/movielens-100k-dataset
    """

    ITEM_COLUMNS = [
        ITEM_ID_COL, 'title', 'release date', 'video release date',
        'IMDb URL ', 'unknown', 'Action', 'Adventure', 'Animation',
        "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
        'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
        'Thriller', 'War', 'Western',
    ]

    @staticmethod
    def load_items(path_to_data: str, item_columns: List[str] = [ITEM_ID_COL, "title"]):
        column_indices = [MovieLens100K.ITEM_COLUMNS.index(col) for col in item_columns]

        return pl.read_csv(
            path_to_data,
            separator='|',
            has_header=False,
            encoding="latin-1",
            dtypes={ITEM_ID_COL: pl.UInt32},
            columns=column_indices,
            new_columns=item_columns
        )

    @staticmethod
    def load_interactions(path_to_data: str, sort: bool = True):
        ratings = pl.read_csv(
            path_to_data,
            separator='\t',
            has_header=False,
            columns=[0, 1, 2],
            new_columns=[USER_ID_COL, ITEM_ID_COL, "rating"],
            dtypes=[pl.UInt32, pl.UInt32, pl.Float32]
        )

        if sort:
            ratings = ratings.sort([USER_ID_COL, ITEM_ID_COL])

        return ratings


class MTSDataset:
    """https://www.kaggle.com/datasets/sharthz23/mts-library
    """

    @staticmethod
    def load_interactions(path_to_data: str):
        interactions = pl.read_csv(path_to_data,
                                   dtypes={
                                       USER_ID_COL: pl.UInt32,
                                       ITEM_ID_COL: pl.UInt32,
                                       "start_date": pl.Date,
                                       "rating": pl.Float32,
                                       "progress": pl.UInt8
                                   }).with_columns(pl.col("rating").fill_null(float("nan")))

        uniq_items = interactions.select(
            pl.col(USER_ID_COL),
            pl.col(ITEM_ID_COL)
        )

        not_duplicated_interactions = interactions.filter(uniq_items.is_unique())
        duplicated_interactions = interactions.filter(uniq_items.is_duplicated()).groupby([USER_ID_COL, ITEM_ID_COL]).agg(
            [
                pl.max("progress"),
                pl.max("rating"),
                pl.min("start_date")
            ]
        )

        return not_duplicated_interactions.vstack(duplicated_interactions)

    @staticmethod
    def load_items(path_to_data: str):
        return pl.read_csv(
            path_to_data,
            dtypes={
                "id": pl.UInt32,
                "genres": pl.Categorical,
                "authors": pl.Categorical,
                "year": pl.Categorical
            },
            new_columns=[ITEM_ID_COL]
        )

    @staticmethod
    def load_users(path_to_data: str):
        return pl.read_csv(path_to_data, dtypes={
            "age": pl.Categorical,
            USER_ID_COL: pl.UInt32,
            "sex": pl.Float32}).with_columns(
                pl.col("sex").cast(pl.Int8)
        )

    @staticmethod
    def select_genres(items: pl.DataFrame, cov: float, null_value: Optional[str] = None):
        """Select genres which covers at most 'cov' examples
        """
        assert 0.0 <= cov <= 1.0, "'cov' must be in range [0; 1]"

        items = items.with_columns(
            pl.col("title").str.strip().str.to_lowercase().alias("title"),
            pl.col("genres").cast(str).str.strip().str.to_lowercase().str.split(
                ",").arr.sort().arr.join(",").alias("genres")
        )

        genres = items.lazy()\
            .filter(pl.col("genres").is_not_null())\
            .select(pl.col("genres").str.split(","))\
            .explode("genres")\
            .groupby("genres")\
            .count()\
            .sort("count", descending=True)\
            .with_columns((pl.col("count") / pl.col("count").sum()).alias("cum_fraction").cumsum())\
            .select(pl.all().exclude("count"))\
            .with_columns((pl.col("genres").cumcount() + 1).alias("num_genres")).collect()

        selected_genres = set(genres.filter(pl.col("cum_fraction") <= cov).select(
            pl.col("genres")).to_series().to_list())

        items = items.with_columns(
            pl.col("genres").str.split(",").apply(lambda x: MTSDataset.exclude_genres(
                x, selected_genres)).alias("genres")
        )

        if null_value is not None:
            items = items.with_columns(pl.col("genres").fill_null(null_value))

        return items

    @staticmethod
    def exclude_genres(genres, selected_genres):
        res = ",".join(sorted(set(genres).intersection(selected_genres)))

        if res:
            return res

        return None


def sample_true_rec_data():
    df_true = pl.DataFrame({
        USER_ID_COL: ['Аня',                'Боря',               'Вася',         'Вася'],
        ITEM_ID_COL: ['Мастер и Маргарита', '451° по Фаренгейту', 'Зеленая миля', 'Рита Хейуорт и спасение из Шоушенка'],
    }
    )

    df_recs = pl.DataFrame({
        USER_ID_COL: [
            'Аня', 'Аня', 'Аня',
            'Боря', 'Боря', 'Боря',
            'Вася', 'Вася', 'Вася',
        ],
        ITEM_ID_COL: [
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
