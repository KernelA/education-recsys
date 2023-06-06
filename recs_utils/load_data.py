from typing import List, Optional

import polars as pl

from .metrics import USER_ID_COL, ITEM_ID_COL


class MovieLens100K:
    """https://www.kaggle.com/datasets/prajitdatta/movielens-100k-dataset
    """
    RELEASE_DATE_COL = "release_date"
    VIDEO_RELEASE_DATE = "video_release_date"

    GENRES_COLUMNS = [
        'unknown',
        'Action',
        'Adventure',
        'Animation',
        "Children's",
        'Comedy',
        'Crime',
        'Documentary',
        'Drama',
        'Fantasy',
        'Film-Noir',
        'Horror',
        'Musical',
        'Mystery',
        'Romance',
        'Sci-Fi',
        'Thriller',
        'War',
        'Western',
    ]

    COL_DTYPES = {
        ITEM_ID_COL: pl.UInt32,
        'title': pl.Utf8,
        RELEASE_DATE_COL: pl.Utf8,
        VIDEO_RELEASE_DATE: pl.Utf8,
        'IMDb_URL': pl.Utf8,
        'unknown': pl.Int8,
        'Action': pl.Int8,
        'Adventure': pl.Int8,
        'Animation': pl.Int8,
        "Children's": pl.Int8,
        'Comedy': pl.Int8,
        'Crime': pl.Int8,
        'Documentary': pl.Int8,
        'Drama': pl.Int8,
        'Fantasy': pl.Int8,
        'Film-Noir': pl.Int8,
        'Horror': pl.Int8,
        'Musical': pl.Int8,
        'Mystery': pl.Int8,
        'Romance': pl.Int8,
        'Sci-Fi': pl.Int8,
        'Thriller': pl.Int8,
        'War': pl.Int8,
        'Western': pl.Int8,
    }

    @staticmethod
    def load_items(path_to_data: str, item_columns: Optional[List[str]] = [ITEM_ID_COL, "title"]):
        if item_columns is None:
            item_columns = list(MovieLens100K.COL_DTYPES.keys())
            column_indices = None
            dtypes = MovieLens100K.COL_DTYPES
        else:
            all_cols = list(MovieLens100K.COL_DTYPES.keys())
            column_indices = [all_cols.index(col) for col in item_columns]
            dtypes = {col: MovieLens100K.COL_DTYPES[col] for col in item_columns}

        data = pl.read_csv(
            path_to_data,
            separator='|',
            has_header=False,
            encoding="latin-1",
            dtypes=dtypes,
            columns=column_indices,
            new_columns=item_columns
        )

        if MovieLens100K.VIDEO_RELEASE_DATE in data.columns:
            data = data.drop(MovieLens100K.VIDEO_RELEASE_DATE)

        actual_genres = [col for col in MovieLens100K.GENRES_COLUMNS if col in data.columns]

        data = data.with_columns(
            pl.concat_list(actual_genres).apply(lambda x: ",".join(
                sorted(actual_genres[pos].lower() for pos, value in enumerate(x) if value == 1))).alias("genres").cast(pl.Categorical)
        )
        data = data.drop(actual_genres)

        if MovieLens100K.RELEASE_DATE_COL in data.columns:
            data = data.with_columns(
                pl.col(MovieLens100K.RELEASE_DATE_COL).str.strptime(pl.Date, "%d-%b-%Y"))

        return data

    @ staticmethod
    def load_users(path_to_data: str):
        return pl.read_csv(
            path_to_data,
            separator="|",
            has_header=False,
            encoding="utf-8",
            new_columns=[USER_ID_COL, "age", "gender", "occupation", "zip_code"],
            dtypes={USER_ID_COL: pl.UInt32, "age": pl.Int16, "gender": pl.Categorical,
                    "occupation": pl.Categorical, "zip_code": pl.Categorical}
        )

    @ staticmethod
    def load_interactions(path_to_data: str, sort: bool = True):
        ratings = pl.read_csv(
            path_to_data,
            separator='\t',
            has_header=False,
            new_columns=[USER_ID_COL, ITEM_ID_COL, "rating", "timestamp"],
            dtypes=[pl.UInt32, pl.UInt32, pl.Float32, pl.Int32]
        )

        ratings = ratings.with_columns(pl.from_epoch(
            pl.col("timestamp"), time_unit="s").dt.date().alias("start_date"))

        ratings = ratings.drop("timestamp")

        if sort:
            ratings = ratings.sort([USER_ID_COL, ITEM_ID_COL])

        return ratings


class MTSDataset:
    """https://www.kaggle.com/datasets/sharthz23/mts-library
    """

    @ staticmethod
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

    @ staticmethod
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

    @ staticmethod
    def load_users(path_to_data: str):
        return pl.read_csv(path_to_data, dtypes={
            "age": pl.Categorical,
            USER_ID_COL: pl.UInt32,
            "sex": pl.Float32}).with_columns(
                pl.col("sex").cast(pl.Int8)
        )

    @ staticmethod
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
            pl.col("genres").str.split(",").apply(lambda x: MTSDataset._exclude_genres(
                x, selected_genres)).alias("genres")
        )

        if null_value is not None:
            items = items.with_columns(pl.col("genres").fill_null(null_value))

        return items

    @ staticmethod
    def _exclude_genres(genres, selected_genres):
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
