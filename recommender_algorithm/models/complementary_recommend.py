
from collections import defaultdict, Counter
from itertools import combinations
import random
import warnings
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim

from base import BaseRecommender


# ---------------------------------------------
# 1. Association Rule Mining (Apriori)
# ---------------------------------------------
class AprioriRecommender(BaseRecommender):
    """Apriori 法による補完推薦を行うクラス。

    トランザクションデータから頻出アイテムセットを抽出し、各アイテムセットを分割して
    ルール（{antecedent} => {consequent}）を抽出する。

    Attributes:
        min_support (float): 最小サポート値。
        min_confidence (float): 最小信頼度。
        rules (List[Tuple[Set[Any], Set[Any], float]]): 推薦ルールのリスト。
        item_support (Dict[frozenset, float]): 各アイテムセットのサポート値。
    """

    def __init__(self, min_support: float = 0.01, min_confidence: float = 0.5) -> None:
        self.min_support: float = min_support
        self.min_confidence: float = min_confidence
        self.rules: List[Tuple[Set[Any], Set[Any], float]] = []
        self.item_support: Dict[frozenset, float] = {}

    def fit(self, transactions: List[List[Any]], **kwargs: Any) -> None:
        """トランザクションデータから頻出アイテムセットおよびルールを抽出する。

        Args:
            transactions: 各取引がアイテムリストとなっているリスト。
            **kwargs: 追加パラメータ（未使用）。
        """
        n_transactions: int = len(transactions)
        support: Dict[frozenset, int] = defaultdict(int)
        for trans in transactions:
            for item in set(trans):
                support[frozenset([item])] += 1

        self.item_support = {itemset: count / n_transactions for itemset, count in support.items()
                             if (count / n_transactions) >= self.min_support}

        current_itemsets: List[frozenset] = list(self.item_support.keys())
        k: int = 2
        while current_itemsets:
            candidate_counts: Dict[frozenset, int] = defaultdict(int)
            for i in range(len(current_itemsets)):
                for j in range(i + 1, len(current_itemsets)):
                    union_set: frozenset = current_itemsets[i] | current_itemsets[j]
                    if len(union_set) == k:
                        candidate_counts[union_set] = 0
            for trans in transactions:
                trans_set: Set[Any] = set(trans)
                for candidate in candidate_counts:
                    if candidate.issubset(trans_set):
                        candidate_counts[candidate] += 1
            current_itemsets = [itemset for itemset, count in candidate_counts.items()
                                if (count / n_transactions) >= self.min_support]
            for itemset in current_itemsets:
                self.item_support[itemset] = candidate_counts[itemset] / n_transactions
            k += 1

        for itemset, support_val in self.item_support.items():
            if len(itemset) < 2:
                continue
            for antecedent in map(frozenset, [list(x) for x in combinations(itemset, len(itemset) - 1)]):
                consequent: frozenset = itemset - antecedent
                if antecedent in self.item_support:
                    confidence: float = support_val / self.item_support[antecedent]
                    if confidence >= self.min_confidence:
                        self.rules.append((set(antecedent), set(consequent), confidence))

    def recommend(self, item: Any, top_k: int = 5) -> List[Any]:
        """指定アイテムを含むルールから補完候補を返す。

        Args:
            item: 基準となるアイテム。
            top_k: 推薦する上位アイテム数。

        Returns:
            推薦候補アイテムのリスト。
        """
        candidates: List[Tuple[Any, float]] = []
        for antecedent, consequent, conf in self.rules:
            if item in antecedent:
                for rec in consequent:
                    candidates.append((rec, conf))
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [itm for itm, _ in candidates[:top_k]]


# ---------------------------------------------
# 2. Co-occurrence Matrix Based
# ---------------------------------------------
class CooccurrenceRecommender(BaseRecommender):
    """アイテム共起行列に基づいて補完推薦を行うクラス。

    Attributes:
        cooccurrence (Dict[Any, Dict[Any, int]]): アイテム間の共起カウント。
        item_counts (Counter): 各アイテムの出現回数。
    """

    def __init__(self) -> None:
        self.cooccurrence: Dict[Any, Dict[Any, int]] = defaultdict(lambda: defaultdict(int))
        self.item_counts: Counter = Counter()

    def fit(self, transactions: List[List[Any]], **kwargs: Any) -> None:
        """トランザクションデータから共起行列を構築する。

        Args:
            transactions: 各取引がアイテムリストとなっているリスト。
            **kwargs: 追加パラメータ（未使用）。
        """
        for trans in transactions:
            unique_items: Set[Any] = set(trans)
            for item in unique_items:
                self.item_counts[item] += 1
            for item1, item2 in combinations(unique_items, 2):
                self.cooccurrence[item1][item2] += 1
                self.cooccurrence[item2][item1] += 1

    def recommend(self, item: Any, top_k: int = 5) -> List[Any]:
        """共起カウントに基づき、指定アイテムと一緒に出現するアイテムを推薦する。

        Args:
            item: 基準となるアイテム。
            top_k: 推薦する上位アイテム数。

        Returns:
            推薦候補アイテムのリスト。
        """
        scores: Dict[Any, int] = self.cooccurrence.get(item, {})
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [itm for itm, _ in sorted_items[:top_k]]


# ---------------------------------------------
# 3. Item2Vec (PyTorch実装)
# ---------------------------------------------
class Item2VecModel(nn.Module):
    """Item2Vec のためのスキップグラムモデル（負例サンプリング付き）。

    Attributes:
        input_embeddings (nn.Embedding): 入力側の埋め込み層。
        output_embeddings (nn.Embedding): 出力側の埋め込み層。
    """

    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        """
        Args:
            vocab_size: アイテム数（語彙数）。
            embedding_dim: 埋め込み次元数。
        """
        super(Item2VecModel, self).__init__()
        self.input_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.output_embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, center: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
        """スキップグラムの損失を計算する。

        Args:
            center: 中心アイテムのインデックステンソル（形状: [batch_size]）。
            pos: 正例（コンテキスト）アイテムのインデックステンソル（形状: [batch_size]）。
            neg: 負例アイテムのインデックステンソル（形状: [batch_size, negative_samples]）。

        Returns:
            損失値（スカラーのテンソル）。
        """
        center_emb = self.input_embeddings(center)            # [B, D]
        pos_emb = self.output_embeddings(pos)                  # [B, D]
        neg_emb = self.output_embeddings(neg)                  # [B, N, D]

        pos_score = torch.sum(center_emb * pos_emb, dim=1)       # [B]
        pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-10)  # [B]

        neg_score = torch.bmm(neg_emb, center_emb.unsqueeze(2)).squeeze(2)  # [B, N]
        neg_loss = -torch.sum(torch.log(torch.sigmoid(-neg_score) + 1e-10), dim=1)  # [B]

        loss = pos_loss + neg_loss
        return loss.mean()


class Item2VecRecommender(BaseRecommender):
    """Item2Vec による補完推薦を行うクラス（PyTorch 実装）。

    各トランザクション内の全ペア (center, context) を正例とし、負例サンプリングを行う。

    Attributes:
        embedding_dim (int): 埋め込み次元数。
        window (int): コンテキストウィンドウサイズ（今回は各取引内全アイテムを対象）。
        negative_samples (int): 負例サンプリング数。
        lr (float): 学習率。
        epochs (int): 学習エポック数。
        batch_size (int): ミニバッチサイズ。
        model (Optional[Item2VecModel]): 学習済みモデル。
        item_to_idx (Dict[Any, int]): アイテム→インデックスのマッピング。
        idx_to_item (Dict[int, Any]): インデックス→アイテムのマッピング。
    """

    def __init__(self, embedding_dim: int = 50, window: int = 2,
                 negative_samples: int = 5, lr: float = 0.01,
                 epochs: int = 10, batch_size: int = 32) -> None:
        self.embedding_dim: int = embedding_dim
        self.window: int = window
        self.negative_samples: int = negative_samples
        self.lr: float = lr
        self.epochs: int = epochs
        self.batch_size: int = batch_size
        self.model: Optional[Item2VecModel] = None
        self.item_to_idx: Dict[Any, int] = {}
        self.idx_to_item: Dict[int, Any] = {}

    def _generate_training_data(self, transactions: List[List[Any]]) -> List[Tuple[int, int]]:
        """トランザクションから正例ペア（center, context）を生成する。

        Args:
            transactions: 各取引がアイテムリストとなっているリスト。

        Returns:
            (center_idx, context_idx) のペアのリスト。
        """
        pairs: List[Tuple[int, int]] = []
        for trans in transactions:
            unique_items = list(set(trans))
            for i in range(len(unique_items)):
                center = unique_items[i]
                for j in range(len(unique_items)):
                    if i == j:
                        continue
                    pairs.append((self.item_to_idx[center], self.item_to_idx[unique_items[j]]))
        return pairs

    def fit(self, transactions: List[List[Any]], **kwargs: Any) -> None:
        """トランザクションデータから語彙を作成し、Item2Vec モデルの学習を行う。

        Args:
            transactions: 各取引がアイテムリストとなっているリスト。
            **kwargs: 追加パラメータ（未使用）。
        """
        vocab: Set[Any] = set()
        for trans in transactions:
            vocab.update(trans)
        self.item_to_idx = {item: idx for idx, item in enumerate(sorted(vocab))}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
        vocab_size: int = len(self.item_to_idx)

        self.model = Item2VecModel(vocab_size, self.embedding_dim)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        pairs: List[Tuple[int, int]] = self._generate_training_data(transactions)
        n_pairs: int = len(pairs)
        if n_pairs == 0:
            return

        for epoch in range(self.epochs):
            random.shuffle(pairs)
            total_loss = 0.0
            for i in range(0, n_pairs, self.batch_size):
                batch = pairs[i:i+self.batch_size]
                center_batch = torch.tensor([p[0] for p in batch], dtype=torch.long)
                pos_batch = torch.tensor([p[1] for p in batch], dtype=torch.long)
                neg_batch = []
                for _ in batch:
                    negatives = []
                    for _ in range(self.negative_samples):
                        neg = random.choice(list(self.item_to_idx.values()))
                        negatives.append(neg)
                    neg_batch.append(negatives)
                neg_batch = torch.tensor(neg_batch, dtype=torch.long)

                optimizer.zero_grad()
                loss = self.model(center_batch, pos_batch, neg_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            # エポック終了時の損失表示（必要に応じて有効化）
            # print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss / (n_pairs / self.batch_size):.4f}")

    def recommend(self, item: Any, top_k: int = 5) -> List[Any]:
        """指定アイテムに類似するアイテムを、学習済み埋め込みの内積から推薦する。

        Args:
            item: 基準となるアイテム。
            top_k: 推薦する上位アイテム数。

        Returns:
            推薦候補アイテムのリスト。
        """
        if item not in self.item_to_idx or self.model is None:
            return []
        self.model.eval()
        with torch.no_grad():
            item_idx = self.item_to_idx[item]
            center_vec = self.model.input_embeddings.weight[item_idx]  # shape: [embedding_dim]
            embeddings = self.model.input_embeddings.weight             # shape: [vocab_size, embedding_dim]
            scores = torch.mv(embeddings, center_vec)                   # shape: [vocab_size]
            scores[item_idx] = -float('inf')  # 自分自身は除外
            
            # 修正：実際の語彙サイズと top_k の小さい方を使用
            actual_top_k = min(top_k, scores.size(0))
            top_values, top_indices = torch.topk(scores, actual_top_k)
            top_indices = top_indices.cpu().numpy()
            return [self.idx_to_item[i] for i in top_indices]

# ---------------------------------------------
# 4. Graph-based (PageRankを利用)
# ---------------------------------------------
class GraphBasedRecommender(BaseRecommender):
    """グラフ上の PageRank を利用して補完推薦を行うクラス。

    Attributes:
        alpha (float): PageRank の減衰係数。
        graph (nx.Graph): アイテム間の共起グラフ。
    """

    def __init__(self, alpha: float = 0.85) -> None:
        self.alpha: float = alpha
        self.graph: nx.Graph = nx.Graph()

    def fit(self, transactions: List[List[Any]], **kwargs: Any) -> None:
        """トランザクションデータからアイテム間の共起グラフを構築する。

        Args:
            transactions: 各取引がアイテムリストとなっているリスト。
            **kwargs: 追加パラメータ（未使用）。
        """
        for trans in transactions:
            unique_items: Set[Any] = set(trans)
            for item in unique_items:
                if not self.graph.has_node(item):
                    self.graph.add_node(item)
            for item1, item2 in combinations(unique_items, 2):
                if self.graph.has_edge(item1, item2):
                    self.graph[item1][item2]['weight'] += 1
                else:
                    self.graph.add_edge(item1, item2, weight=1)

    def recommend(self, item: Any, top_k: int = 5) -> List[Any]:
        """パーソナライズド PageRank により指定アイテムと関連性の高いアイテムを推薦する。

        Args:
            item: 基準となるアイテム。
            top_k: 推薦する上位アイテム数。

        Returns:
            推薦候補アイテムのリスト。
        """
        if item not in self.graph:
            return []
        personalization: Dict[Any, float] = {node: 0.0 for node in self.graph.nodes()}
        personalization[item] = 1.0
        pr: Dict[Any, float] = nx.pagerank(self.graph, alpha=self.alpha,
                                            personalization=personalization,
                                            weight='weight')
        pr.pop(item, None)
        sorted_pr = sorted(pr.items(), key=lambda x: x[1], reverse=True)
        return [node for node, _ in sorted_pr[:top_k]]


# ---------------------------------------------
# 5. Bayesian Personalized Ranking (BPR)
# ---------------------------------------------
class BPRRecommender(BaseRecommender):
    """BPR (Bayesian Personalized Ranking) を用いて補完推薦を行うクラス。

    Attributes:
        latent_dim (int): 潜在ベクトルの次元数。
        learning_rate (float): 学習率。
        epochs (int): 学習エポック数。
        reg (float): 正則化係数。
        item_factors (Dict[Any, np.ndarray]): 各アイテムの潜在ベクトル。
        items (List[Any]): 全アイテムリスト。
    """

    def __init__(self, latent_dim: int = 20, learning_rate: float = 0.01,
                 epochs: int = 10, reg: float = 0.01) -> None:
        self.latent_dim: int = latent_dim
        self.learning_rate: float = learning_rate
        self.epochs: int = epochs
        self.reg: float = reg
        self.item_factors: Dict[Any, np.ndarray] = {}
        self.items: Optional[List[Any]] = None

    def fit(self, transactions: List[List[Any]], **kwargs: Any) -> None:
        """BPR の学習を行う。

        Args:
            transactions: 各取引がアイテムリストとなっているリスト。
            **kwargs: 追加パラメータ（未使用）。
        """
        item_set: Set[Any] = set()
        for trans in transactions:
            item_set.update(trans)
        self.items = list(item_set)
        for item in self.items:
            self.item_factors[item] = np.random.normal(scale=0.1, size=self.latent_dim)
        positive_pairs: List[Tuple[Any, Any]] = []
        for trans in transactions:
            for item1, item2 in combinations(set(trans), 2):
                positive_pairs.append((item1, item2))
                positive_pairs.append((item2, item1))
        for epoch in range(self.epochs):
            random.shuffle(positive_pairs)
            for (i, j) in positive_pairs:
                k: Any = random.choice(self.items)
                while k == i or k == j:
                    k = random.choice(self.items)
                x_ij: float = np.dot(self.item_factors[i], self.item_factors[j])
                x_ik: float = np.dot(self.item_factors[i], self.item_factors[k])
                x_diff: float = x_ij - x_ik
                sigmoid: float = 1 / (1 + np.exp(-x_diff))
                grad: float = self.learning_rate * (1 - sigmoid)
                self.item_factors[i] += grad * (self.item_factors[j] - self.item_factors[k]) - self.reg * self.item_factors[i]
                self.item_factors[j] += grad * self.item_factors[i] - self.reg * self.item_factors[j]
                self.item_factors[k] -= grad * self.item_factors[i] - self.reg * self.item_factors[k]

    def recommend(self, item: Any, top_k: int = 5) -> List[Any]:
        """指定アイテムの潜在ベクトルと内積が高いアイテムを推薦する。

        Args:
            item: 基準となるアイテム。
            top_k: 推薦する上位アイテム数。

        Returns:
            推薦候補アイテムのリスト。
        """
        if item not in self.item_factors or self.items is None:
            return []
        query_vec: np.ndarray = self.item_factors[item]
        scores: Dict[Any, float] = {}
        for other in self.items:
            if other == item:
                continue
            scores[other] = np.dot(query_vec, self.item_factors[other])
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [itm for itm, _ in sorted_scores[:top_k]]


# ---------------------------------------------
# 6. Conditional Probability Based
# ---------------------------------------------
class ConditionalProbabilityRecommender(BaseRecommender):
    """条件付き確率に基づく補完推薦を行うクラス。

    Attributes:
        item_counts (Counter): 各アイテムの出現回数。
        pair_counts (Dict[Any, Dict[Any, int]]): アイテムペアの同時出現カウント。
    """

    def __init__(self) -> None:
        self.item_counts: Counter = Counter()
        self.pair_counts: Dict[Any, Dict[Any, int]] = defaultdict(lambda: defaultdict(int))

    def fit(self, transactions: List[List[Any]], **kwargs: Any) -> None:
        """トランザクションデータから出現カウントを計算する。

        Args:
            transactions: 各取引がアイテムリストとなっているリスト。
            **kwargs: 追加パラメータ（未使用）。
        """
        for trans in transactions:
            unique: Set[Any] = set(trans)
            for item in unique:
                self.item_counts[item] += 1
            for i, j in combinations(unique, 2):
                self.pair_counts[i][j] += 1
                self.pair_counts[j][i] += 1

    def recommend(self, item: Any, top_k: int = 5) -> List[Any]:
        """P(other|item) = count(item, other) / count(item) の高いアイテムを推薦する。

        Args:
            item: 基準となるアイテム。
            top_k: 推薦する上位アイテム数。

        Returns:
            推薦候補アイテムのリスト。
        """
        candidates: Dict[Any, float] = {}
        for other, count in self.pair_counts.get(item, {}).items():
            candidates[other] = count / self.item_counts[item]
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        return [itm for itm, _ in sorted_candidates[:top_k]]


# ---------------------------------------------
# 7. Category-Based Complementary
# ---------------------------------------------
class CategoryBasedRecommender(BaseRecommender):
    """カテゴリ情報を利用した補完推薦を行うクラス。

    Attributes:
        item_categories (Dict[Any, str]): 各アイテムのカテゴリ。
        category_item_map (Dict[str, List[Any]]): カテゴリごとのアイテムリスト。
        item_cooccurrence (Dict[Any, Dict[Any, int]]): アイテム間の共起カウント。
    """

    def __init__(self) -> None:
        self.item_categories: Dict[Any, str] = {}
        self.category_item_map: Dict[str, List[Any]] = defaultdict(list)
        self.item_cooccurrence: Dict[Any, Dict[Any, int]] = defaultdict(lambda: defaultdict(int))

    def fit(self, transactions: List[List[Any]], item_categories: Dict[Any, str], **kwargs: Any) -> None:
        """トランザクションとアイテムカテゴリ情報から補完推薦のためのデータを構築する。

        Args:
            transactions: 各取引がアイテムリストとなっているリスト。
            item_categories: アイテム→カテゴリのマッピング辞書。
            **kwargs: 追加パラメータ（未使用）。
        """
        self.item_categories = item_categories
        for item, cat in item_categories.items():
            self.category_item_map[cat].append(item)
        for trans in transactions:
            unique: Set[Any] = set(trans)
            for i, j in combinations(unique, 2):
                self.item_cooccurrence[i][j] += 1
                self.item_cooccurrence[j][i] += 1

    def recommend(self, item: Any, top_k: int = 5) -> List[Any]:
        """同一取引中の共起情報とカテゴリ情報を組み合わせ、補完候補を推薦する。

        Args:
            item: 基準となるアイテム。
            top_k: 推薦する上位アイテム数。

        Returns:
            推薦候補アイテムのリスト。
        """
        if item not in self.item_categories:
            return []
        query_cat: str = self.item_categories[item]
        cat_scores: Dict[str, int] = defaultdict(int)
        for other, count in self.item_cooccurrence.get(item, {}).items():
            other_cat: Optional[str] = self.item_categories.get(other)
            if other_cat is None or other_cat == query_cat:
                continue
            cat_scores[other_cat] += count
        if not cat_scores:
            return []
        target_cat: str = max(cat_scores.items(), key=lambda x: x[1])[0]
        candidates: Dict[Any, int] = {}
        for candidate in self.category_item_map[target_cat]:
            if candidate == item:
                continue
            candidates[candidate] = self.item_cooccurrence.get(item, {}).get(candidate, 0)
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        return [itm for itm, _ in sorted_candidates[:top_k]]


# ---------------------------------------------
# 8. Rule-Based Cross-Sell
# ---------------------------------------------
class RuleBasedCrossSellRecommender(BaseRecommender):
    """事前定義ルールに基づく補完推薦を行うクラス。

    Attributes:
        rules (Dict[Any, List[Any]]): アイテムごとの推薦ルール辞書。
    """

    def __init__(self, rules: Optional[Dict[Any, List[Any]]] = None) -> None:
        self.rules: Dict[Any, List[Any]] = rules if rules is not None else {}

    def fit(self, data: Any, **kwargs: Any) -> None:
        """ルール辞書を設定する。

        Args:
            data: ルール辞書（例: {'A': ['B', 'C'], ...}）。
            **kwargs: 追加パラメータ（未使用）。
        """
        if isinstance(data, dict):
            self.rules = data

    def recommend(self, item: Any, top_k: int = 5) -> List[Any]:
        """事前定義されたルールに基づき推薦を返す。

        Args:
            item: 基準となるアイテム。
            top_k: 推薦する上位アイテム数。

        Returns:
            推薦候補アイテムのリスト。
        """
        recs: List[Any] = self.rules.get(item, [])
        return recs[:top_k]


# ---------------------------------------------
# 9. Neural Network-based (Multi-task Learning)
# ---------------------------------------------
class NeuralComplementaryRecommender(BaseRecommender):
    """ニューラルネットワークを用いて補完推薦を学習するクラス。

    Attributes:
        num_items (int): 全アイテム数（初期設定用）。
        latent_dim (int): 埋め込み次元数。
        lr (float): 学習率。
        epochs (int): 学習エポック数。
        model (Optional[nn.Module]): 学習済みモデル。
        item_to_idx (Dict[Any, int]): アイテム→インデックスのマッピング。
        idx_to_item (Dict[int, Any]): インデックス→アイテムのマッピング。
    """

    def __init__(self, num_items: int, latent_dim: int = 32,
                 lr: float = 0.001, epochs: int = 5) -> None:
        self.num_items: int = num_items
        self.latent_dim: int = latent_dim
        self.lr: float = lr
        self.epochs: int = epochs
        self.model: Optional[nn.Module] = None
        self.item_to_idx: Dict[Any, int] = {}
        self.idx_to_item: Dict[int, Any] = {}

    def _build_model(self) -> nn.Module:
        """ニューラルネットワークモデルを構築する。

        Returns:
            構築された nn.Module モデル。
        """
        class SimpleNN(nn.Module):
            def __init__(self, num_items: int, latent_dim: int) -> None:
                super(SimpleNN, self).__init__()
                self.embedding = nn.Embedding(num_items, latent_dim)
                self.fc = nn.Linear(latent_dim, num_items)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                emb = self.embedding(x)
                out = self.fc(emb)
                return out

        return SimpleNN(self.num_items, self.latent_dim)

    def fit(self, transactions: List[List[Any]], **kwargs: Any) -> None:
        """トランザクションから (入力, ターゲット) ペアを作成し、ニューラルネットワークを学習する。

        Args:
            transactions: 各取引がアイテムリストとなっているリスト。
            **kwargs: 追加パラメータ（未使用）。
        """
        items: Set[Any] = set()
        for trans in transactions:
            items.update(trans)
        self.item_to_idx = {item: idx for idx, item in enumerate(sorted(items))}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
        self.num_items = len(self.item_to_idx)

        self.model = self._build_model()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()

        pairs: List[Tuple[int, int]] = []
        for trans in transactions:
            unique = list(set(trans))
            for i in unique:
                for j in unique:
                    if i != j:
                        pairs.append((self.item_to_idx[i], self.item_to_idx[j]))
        if not pairs:
            return
        inputs = torch.tensor([p[0] for p in pairs], dtype=torch.long)
        targets = torch.tensor([p[1] for p in pairs], dtype=torch.long)

        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

    def recommend(self, item: Any, top_k: int = 5) -> List[Any]:
        """指定アイテムに対してニューラルネットワークから推薦を行う。

        Args:
            item: 基準となるアイテム。
            top_k: 推薦する上位アイテム数。

        Returns:
            推薦候補アイテムのリスト。
        """
        if item not in self.item_to_idx or self.model is None:
            return []
        self.model.eval()
        with torch.no_grad():
            idx = torch.tensor([self.item_to_idx[item]], dtype=torch.long)
            logits = self.model(idx)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
            probs[self.item_to_idx[item]] = -1
            top_indices = np.argsort(probs)[-top_k:][::-1]
            return [self.idx_to_item[i] for i in top_indices]


# ---------------------------------------------
# 10. MF + Co-occurrence Features
# ---------------------------------------------
class MFCooccurrenceRecommender(BaseRecommender):
    """行列分解と共起特徴を組み合わせた補完推薦を行うクラス。

    Attributes:
        latent_dim (int): 潜在ベクトルの次元数。
        lr (float): 学習率。
        epochs (int): 学習エポック数。
        alpha (float): MFスコアと共起スコアの重み係数。
        item_factors (Dict[Any, np.ndarray]): 各アイテムの潜在ベクトル。
        items (List[Any]): 全アイテムリスト。
        cooccurrence (Dict[Any, Dict[Any, int]]): アイテム間の共起カウント。
    """

    def __init__(self, latent_dim: int = 20, lr: float = 0.01, epochs: int = 10, alpha: float = 0.5) -> None:
        self.latent_dim: int = latent_dim
        self.lr: float = lr
        self.epochs: int = epochs
        self.alpha: float = alpha
        self.item_factors: Dict[Any, np.ndarray] = {}
        self.items: Optional[List[Any]] = None
        self.cooccurrence: Dict[Any, Dict[Any, int]] = defaultdict(lambda: defaultdict(int))

    def fit(self, transactions: List[List[Any]], **kwargs: Any) -> None:
        """トランザクションデータから共起行列と行列分解の学習を行う。

        Args:
            transactions: 各取引がアイテムリストとなっているリスト。
            **kwargs: 追加パラメータ（未使用）。
        """
        for trans in transactions:
            unique = set(trans)
            for i in unique:
                for j in unique:
                    if i != j:
                        self.cooccurrence[i][j] += 1
        item_set: Set[Any] = set()
        for trans in transactions:
            item_set.update(trans)
        self.items = list(item_set)
        for item in self.items:
            self.item_factors[item] = np.random.normal(scale=0.1, size=self.latent_dim)
        pairs: List[Tuple[Any, Any]] = []
        for trans in transactions:
            for i, j in combinations(set(trans), 2):
                pairs.append((i, j))
                pairs.append((j, i))
        for epoch in range(self.epochs):
            random.shuffle(pairs)
            for (i, j) in pairs:
                k: Any = random.choice(self.items)
                while k == i or k == j:
                    k = random.choice(self.items)
                x_ij: float = np.dot(self.item_factors[i], self.item_factors[j])
                x_ik: float = np.dot(self.item_factors[i], self.item_factors[k])
                x_diff: float = x_ij - x_ik
                sigmoid: float = 1 / (1 + np.exp(-x_diff))
                grad: float = self.lr * (1 - sigmoid)
                self.item_factors[i] += grad * (self.item_factors[j] - self.item_factors[k]) - self.lr * self.item_factors[i]
                self.item_factors[j] += grad * self.item_factors[i] - self.lr * self.item_factors[j]
                self.item_factors[k] -= grad * self.item_factors[i] - self.lr * self.item_factors[k]

    def recommend(self, item: Any, top_k: int = 5) -> List[Any]:
        """指定アイテムに対して、MFスコアと共起スコアを組み合わせたランキングで推薦する。

        Args:
            item: 基準となるアイテム。
            top_k: 推薦する上位アイテム数。

        Returns:
            推薦候補アイテムのリスト。
        """
        if item not in self.item_factors or self.items is None:
            return []
        scores: Dict[Any, float] = {}
        co_scores: Dict[Any, int] = self.cooccurrence.get(item, {})
        for other in self.items:
            if other == item:
                continue
            mf_score: float = np.dot(self.item_factors[item], self.item_factors[other])
            co_score: int = co_scores.get(other, 0)
            score: float = mf_score + self.alpha * co_score
            scores[other] = score
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [itm for itm, _ in sorted_scores[:top_k]]


# ---------------------------------------------
# テスト用サンプル（任意）
# ---------------------------------------------
if __name__ == "__main__":
    # サンプルデータ: 各取引はアイテムリスト（例: 商品ID）
    transactions: List[List[str]] = [
        ['A', 'B', 'C'],
        ['A', 'C'],
        ['B', 'D'],
        ['A', 'D'],
        ['B', 'C', 'D'],
        ['C', 'D']
    ]
    # アイテムカテゴリ情報（Category-Based 用）
    item_categories: Dict[str, str] = {
        'A': 'Electronics',
        'B': 'Accessories',
        'C': 'Electronics',
        'D': 'Home'
    }
    # ルールベース用ルール
    rules: Dict[str, List[str]] = {
        'A': ['D'],
        'B': ['C'],
    }

    print("== AprioriRecommender ==")
    apriori_rec = AprioriRecommender(min_support=0.2, min_confidence=0.5)
    apriori_rec.fit(transactions)
    print("Recommend for 'A':", apriori_rec.recommend('A'))

    print("\n== CooccurrenceRecommender ==")
    coocc_rec = CooccurrenceRecommender()
    coocc_rec.fit(transactions)
    print("Recommend for 'A':", coocc_rec.recommend('A'))

    print("\n== Item2VecRecommender (PyTorch実装) ==")
    item2vec_rec = Item2VecRecommender(embedding_dim=10, window=2, negative_samples=5, lr=0.01, epochs=20, batch_size=4)
    item2vec_rec.fit(transactions)
    print("Recommend for 'A':", item2vec_rec.recommend('A'))

    print("\n== GraphBasedRecommender ==")
    graph_rec = GraphBasedRecommender(alpha=0.85)
    graph_rec.fit(transactions)
    print("Recommend for 'A':", graph_rec.recommend('A'))

    print("\n== BPRRecommender ==")
    bpr_rec = BPRRecommender(latent_dim=10, learning_rate=0.01, epochs=20, reg=0.01)
    bpr_rec.fit(transactions)
    print("Recommend for 'A':", bpr_rec.recommend('A'))

    print("\n== ConditionalProbabilityRecommender ==")
    cond_rec = ConditionalProbabilityRecommender()
    cond_rec.fit(transactions)
    print("Recommend for 'A':", cond_rec.recommend('A'))

    print("\n== CategoryBasedRecommender ==")
    cat_rec = CategoryBasedRecommender()
    cat_rec.fit(transactions, item_categories=item_categories)
    print("Recommend for 'A':", cat_rec.recommend('A'))

    print("\n== RuleBasedCrossSellRecommender ==")
    rule_rec = RuleBasedCrossSellRecommender(rules=rules)
    rule_rec.fit(rules)
    print("Recommend for 'A':", rule_rec.recommend('A'))

    print("\n== NeuralComplementaryRecommender ==")
    neural_rec = NeuralComplementaryRecommender(num_items=10, latent_dim=8, lr=0.005, epochs=10)
    neural_rec.fit(transactions)
    print("Recommend for 'A':", neural_rec.recommend('A'))

    print("\n== MFCooccurrenceRecommender ==")
    mf_co_rec = MFCooccurrenceRecommender(latent_dim=10, lr=0.01, epochs=10, alpha=0.5)
    mf_co_rec.fit(transactions)
    print("Recommend for 'A':", mf_co_rec.recommend('A'))