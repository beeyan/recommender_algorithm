import abc

class BaseRecommender(abc.ABC):
    """
    レコメンドアルゴリズム共通の抽象クラス。
    """

    @abc.abstractmethod
    def fit(self, user_item_data):
        """モデルの学習を行う。

        Args:
            data: 学習に使用するデータ（例: トランザクションのリスト）。
            **kwargs: その他のオプションパラメータ。
        """
        pass

    @abc.abstractmethod
    def recommend(self, user_id, top_k=5):
        """推薦候補を生成する。

        Args:
            query: 推薦の基準となるアイテムまたはユーザー。
            top_k: 推薦する上位アイテム数。

        Returns:
            推薦されたアイテムのリスト。
        """
        pass

    # オプション: 評価指標計算 (Precision@K, Recall@K など)
    def evaluate(self, val_data):
        pass