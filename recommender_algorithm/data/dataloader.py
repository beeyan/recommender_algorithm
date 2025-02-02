import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """
    CSVファイルからデータを読み込み、DataFrameを返す。
    """
    df = pd.read_csv(path)
    return df


def split_data(df: pd.DataFrame, split_ratio: float = 0.8):
    """
    ユーザー-アイテムデータを学習用と評価用に分割。
    :param df: 入力DataFrame
    :param split_ratio: 学習データに割り当てる比率
    :return: (train_df, val_df)
    """
    # シンプルなサンプルとしてランダムシャッフル & 分割
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_index = int(len(df_shuffled) * split_ratio)
    train_df = df_shuffled[:split_index]
    val_df = df_shuffled[split_index:]
    return train_df, val_df