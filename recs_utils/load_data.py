import polars as pl

from .metrics import USER_ID_COL, ITEM_ID_COL


class MovieLens100K:
    ITEM_COLUMNS = [
        ITEM_ID_COL, 'title', 'release date', 'video release date',
        'IMDb URL ', 'unknown', 'Action', 'Adventure', 'Animation',
        "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
        'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
        'Thriller', 'War', 'Western',
    ]

    @staticmethod
    def load(path_to_date: str, path_to_items: str, item_num_read_columns: int = 2):
        ratings = pl.read_csv(
            path_to_date,
            separator='\t',
            has_header=False,
            columns=[0, 1, 2],
            new_columns=[USER_ID_COL, ITEM_ID_COL, "rating"],
            dtypes=[pl.UInt32, pl.UInt32, pl.Float32]
        ).sort([USER_ID_COL, ITEM_ID_COL])

        item_num_read_columns = min(len(MovieLens100K.ITEM_COLUMNS), item_num_read_columns)

        movies = pl.read_csv(
            path_to_items,
            separator='|',
            has_header=False,
            encoding="latin-1",
            dtypes={ITEM_ID_COL: pl.UInt32},
            columns=list(range(item_num_read_columns)),
            new_columns=MovieLens100K.ITEM_COLUMNS[:item_num_read_columns],
        )

        return ratings, movies


def load_users(path_to_file: str):
    df_users = pl.read_csv(path_to_file, dtypes={
        "age": pl.Categorical, USER_ID_COL: pl.UInt32, "sex": pl.Float32})
    return df_users


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


def load_interactions(path_to_file: str):
    interactions = pl.read_csv(path_to_file,
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


def load_items(path_to_file: str):
    return pl.read_csv(
        path_to_file,
        dtypes={
            "id": pl.UInt32,
            "genres": pl.Categorical,
            "authors": pl.Categorical,
            "year": pl.Categorical
        },
        new_columns=[ITEM_ID_COL]
    )
