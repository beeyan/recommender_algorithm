import pandas as pd


def encode_user_item_ids(df: pd.DataFrame, user_col: str = "user_id", item_col: str = "item_id"):
    """
    ユーザーID、アイテムIDを連番に変換するなど、特徴量エンジニアリングに利用する。
    """
    user_ids = df[user_col].unique()
    item_ids = df[item_col].unique()

    user2idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
    item2idx = {item_id: idx for idx, item_id in enumerate(item_ids)}

    df["user_idx"] = df[user_col].map(user2idx)
    df["item_idx"] = df[item_col].map(item2idx)

    return df, user2idx, item2idx