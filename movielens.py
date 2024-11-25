import pandas as pd
import os


def load_pandas_df(
    data_path="100k/",
    header=None,
    title_col=None,
    genres_col=None,
    year_col=None,
):
    """
    Loads the MovieLens dataset as a pandas DataFrame from local files.

    Args:
        data_path (str): Path to the MovieLens dataset directory (default is `/ml-100k/`).
        header (list or tuple or None): Rating dataset header.
            If None, defaults to ["UserId", "ItemId", "Rating", "Timestamp"].
        title_col (str): Movie title column name. If None, the column will not be loaded.
        genres_col (str): Genres column name. Genres are '|' separated string.
            If None, the column will not be loaded.
        year_col (str): Movie release year column name. If None, the column will not be loaded.

    Returns:
        pandas.DataFrame: Combined ratings and movie information DataFrame.
    """
    # Set default header if not provided
    if header is None:
        header = ["UserId", "ItemId", "Rating", "Timestamp"]

    # File paths
    ratings_file = os.path.join(data_path, "u.data")
    items_file = os.path.join(data_path, "u.item")

    # Load ratings
    ratings = pd.read_csv(
        ratings_file,
        sep="\t",
        names=header,
        engine="python",
    )

    # Load movie metadata
    item_cols = ["ItemId", "Title", "ReleaseDate", "VideoReleaseDate", "IMDbURL"] + [
        f"Genre_{i}" for i in range(19)
    ]
    items = pd.read_csv(
        items_file,
        sep="|",
        names=item_cols,
        encoding="ISO-8859-1",
        engine="python",
    )

    # Select only relevant columns from the items dataset
    selected_cols = ["ItemId"]
    if title_col:
        selected_cols.append("Title")
    if genres_col:
        selected_cols += [f"Genre_{i}" for i in range(19)]
    if year_col:
        items["Year"] = items["ReleaseDate"].str[-4:]  # Extract year from ReleaseDate
        selected_cols.append("Year")

    items = items[selected_cols]

    # Merge ratings with movie metadata
    df = ratings.merge(items, on="ItemId", how="left")

    return df
