import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# probably not worth using this
# def multihot(
#     df: pd.DataFrame, genre: bool = False, keyword: bool = False, tagline: bool = False
# ) -> pd.DataFrame:
#     if genre:
#         df = multihot_column(df, delim="-", column="genres")
#     if keyword:
#         df = multihot_tf_idf(df, "keywords")
#     if tagline:
#         pass
#     return df


def multihot_column(df: pd.DataFrame, delim: str, column: str) -> pd.DataFrame:
    column_set = set()
    df = df.dropna(subset=[column])
    for i in df[column].dropna():
        for j in i.split(delim):
            column_set.add(j)

    for col_val in column_set:
        df[": ".join([column, col_val])] = df.apply(
            lambda row: 1 if col_val in row[column] else 0, axis=1
        )

    df = df.drop(columns=[column])

    return df


def multihot_tf_idf(df: pd.DataFrame, column: str):
    df = df.dropna(subset=[column])
    feature_set = tf_idf_keywords(df, "-", column)
    for val in feature_set:
        df[": ".join([column, val])] = df.apply(
            lambda row: 1 if val in row[column] else 0, axis=1
        )
    df = df.drop(columns=[column])
    return df


def tf_idf_keywords(
    df: pd.DataFrame,
    delim: str,
    column: str,
    max_features: int = 1000,
    num_terms: int = 50,
):
    df[column] = df[column].apply(lambda string: string.replace(" ", "_"))
    df[column] = df[column].apply(lambda string: " ".join(string.split(delim)))
    tfidf = TfidfVectorizer(max_features=max_features, stop_words="english")
    tf_idf_out = tfidf.fit_transform(df[column])
    terms = tfidf.get_feature_names_out()
    top_indices = np.argsort(-tf_idf_out.sum(axis=0))[:, :num_terms]  # type: ignore
    top_terms = [terms[i] for i in top_indices][0][0]
    return [i.replace("_", " ") for i in top_terms]


def profit_margin(cost: pd.Series, revenue: pd.Series):
    return (revenue - cost) / revenue
