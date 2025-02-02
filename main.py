import os
import pandas as pd

from recommender_algorithm.data.dataloader import load_data, split_data
from recommender_algorithm.features.create_features import encode_user_item_ids
from recommender_algorithm.models.simple_recommender import PopularityRecommender
from recommender_algorithm.models.collaborative_filtering import UserBasedCFRecommender
from recommender_algorithm.utils.logger import get_logger
from recommender_algorithm.utils.metrics import precision_at_k, recall_at_k


def main():
    logger = get_logger(name="recommender_example")

    # 1. データ読み込み
    data_path = os.path.join("data", "user_item_rating.csv")
    df = load_data(data_path)
    logger.info(f"Loaded data shape: {df.shape}")

    # 2. 前処理 & 分割
    train_df, val_df = split_data(df, split_ratio=0.8)
    logger.info(f"Train data shape: {train_df.shape}, Val data shape: {val_df.shape}")

    # 3. 特徴量エンジニアリング（ユーザーID, アイテムIDのエンコード例など）
    # 実際にはエンコード済みデータを使うかどうかはユースケース次第
    train_df, user2idx, item2idx = encode_user_item_ids(train_df)
    val_df, _, _ = encode_user_item_ids(val_df)

    # 4. モデル定義 & 学習（例: 人気ベース）
    pop_rec = PopularityRecommender()
    pop_rec.fit(train_df)

    # 5. モデル定義 & 学習（例: ユーザーベースCF）
    cf_rec = UserBasedCFRecommender(similarity_measure="cosine", k_neighbors=10)
    cf_rec.fit(train_df)

    # 6. 評価 (サンプルとして1ユーザーに対してPrecision@K, Recall@Kを計算)
    sample_user = train_df["user_id"].iloc[0]
    # 実際には評価データにいるユーザーを選ぶことを想定
    # バリデーションデータでユーザーが選んだアイテムを正解セットとする
    user_val_items = val_df[val_df["user_id"] == sample_user]["item_id"].unique().tolist()

    if user_val_items:
        # 人気ベースでレコメンド
        pop_predictions = pop_rec.recommend(sample_user, top_k=5)
        precision_pop = precision_at_k(pop_predictions, set(user_val_items), 5)
        recall_pop = recall_at_k(pop_predictions, set(user_val_items), 5)
        logger.info(f"[Popularity] Precision@5: {precision_pop}, Recall@5: {recall_pop}")

        # ユーザーベースCFでレコメンド
        cf_predictions = cf_rec.recommend(sample_user, top_k=5)
        precision_cf = precision_at_k(cf_predictions, set(user_val_items), 5)
        recall_cf = recall_at_k(cf_predictions, set(user_val_items), 5)
        logger.info(f"[UserBasedCF] Precision@5: {precision_cf}, Recall@5: {recall_cf}")

    # 7. レコメンド例 (任意のユーザーIDで)
    recommended_items_pop = pop_rec.recommend(user_id=999, top_k=5)
    logger.info(f"Recommended items (Popularity): {recommended_items_pop}")
    recommended_items_cf = cf_rec.recommend(user_id=999, top_k=5)
    logger.info(f"Recommended items (UserBasedCF): {recommended_items_cf}")


if __name__ == "__main__":
    main()
