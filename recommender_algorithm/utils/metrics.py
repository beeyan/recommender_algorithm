# metrics.py

from math import log2
from typing import Any, List, Set, Sequence


def precision_at_k(recommended: List[Any], relevant: Set[Any], k: int) -> float:
    """Precision@k を計算する。

    推薦された上位 k 件中、実際に正解となるアイテムの割合を返す。

    Args:
        recommended: 推薦されたアイテムのリスト。
        relevant: 正解（関連性のある）アイテムの集合。
        k: 評価対象の上位件数。

    Returns:
        Precision@k の値（0.0～1.0）。
    """
    recommended_k = recommended[:k]
    if not recommended_k:
        return 0.0
    return len(set(recommended_k) & relevant) / float(k)


def recall_at_k(recommended: List[Any], relevant: Set[Any], k: int) -> float:
    """Recall@k を計算する。

    推薦された上位 k 件中、実際の正解アイテムがどの程度含まれているかを返す。

    Args:
        recommended: 推薦されたアイテムのリスト。
        relevant: 正解（関連性のある）アイテムの集合。
        k: 評価対象の上位件数。

    Returns:
        Recall@k の値（0.0～1.0）。正解アイテム数が 0 の場合は 0.0 を返す。
    """
    if not relevant:
        return 0.0
    recommended_k = recommended[:k]
    return len(set(recommended_k) & relevant) / float(len(relevant))


def mrr_at_k(recommended: List[Any], relevant: Set[Any], k: int) -> float:
    """MRR@k（Mean Reciprocal Rank）を計算する。

    推薦順位のうち、最初に正解となるアイテムが現れる順位の逆数を返す。

    Args:
        recommended: 推薦されたアイテムのリスト。
        relevant: 正解（関連性のある）アイテムの集合。
        k: 評価対象の上位件数。

    Returns:
        MRR@k の値（0.0～1.0）。上位 k 件に正解がなければ 0.0 を返す。
    """
    for idx, item in enumerate(recommended[:k]):
        if item in relevant:
            return 1.0 / (idx + 1)
    return 0.0


def average_precision_at_k(recommended: List[Any], relevant: Set[Any], k: int) -> float:
    """Average Precision@k (AP@k) を計算する。

    推薦された上位 k 件それぞれにおける Precision の平均値を返す。

    Args:
        recommended: 推薦されたアイテムのリスト。
        relevant: 正解（関連性のある）アイテムの集合。
        k: 評価対象の上位件数。

    Returns:
        AP@k の値（0.0～1.0）。正解アイテムが存在しない場合は 0.0 を返す。
    """
    recommended_k = recommended[:k]
    if not relevant:
        return 0.0
    score = 0.0
    num_hits = 0.0
    for i, item in enumerate(recommended_k):
        if item in relevant:
            num_hits += 1.0
            score += num_hits / (i + 1)
    return score / float(len(relevant))


def ndcg_at_k(recommended: List[Any], relevant: Set[Any], k: int) -> float:
    """nDCG@k (Normalized Discounted Cumulative Gain) を計算する。

    Args:
        recommended: 推薦されたアイテムのリスト。
        relevant: 正解（関連性のある）アイテムの集合。関連性は二値（1: 正解, 0: 非正解）とする。
        k: 評価対象の上位件数。

    Returns:
        nDCG@k の値（0.0～1.0）。正解アイテムが存在しない場合は 0.0 を返す。
    """
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        rel = 1.0 if item in relevant else 0.0
        dcg += rel / log2(i + 2)  # log2(i+2) は i=0 の場合 log2(2)=1
    # 理想的なDCG (IDCG)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0.0 else 0.0

def hit_rate_at_k(recommended: List[Any], relevant: Set[Any], k: int) -> float:
    """Hit Rate@k を計算する関数。

    推薦された上位 k 件の中に、少なくとも1件の正解アイテムが含まれていれば 1.0、含まれていなければ 0.0 を返す。

    Args:
        recommended: 推薦されたアイテムのリスト。
        relevant: 正解（関連性のある）アイテムの集合。
        k: 評価対象とする上位件数。

    Returns:
        Hit Rate@k の値（0.0 または 1.0）。
    """
    recommended_k = recommended[:k]
    return 1.0 if set(recommended_k) & relevant else 0.0


def r_precision(recommended: List[Any], relevant: Set[Any]) -> float:
    """R-Precision を計算する関数。

    R-Precision は、正解アイテム数 R を上位 R 件での精度として計算する指標である。
    正解アイテム数 R が 0 の場合は 0.0 を返す。

    Args:
        recommended: 推薦されたアイテムのリスト。
        relevant: 正解（関連性のある）アイテムの集合。

    Returns:
        R-Precision の値（0.0～1.0）。
    """
    R = len(relevant)
    if R == 0:
        return 0.0
    recommended_R = recommended[:R]
    return len(set(recommended_R) & relevant) / float(R)


def auc_score(ranking: List[Any], relevant: Set[Any]) -> float:
    """AUC (Area Under the ROC Curve) を計算する関数。

    完全なランキングリストに対して、正例の順位に基づく AUC を計算する。
    ランキングは 1-indexed として、正解アイテムの順位の合計から理想順位の合計を引き、
    正例数と負例数の積で割ることで求める。

    Args:
        ranking: すべてのアイテムを順位順に並べたリスト。
        relevant: 正解（関連性のある）アイテムの集合。

    Returns:
        AUC の値（0.0～1.0）。正例または負例が存在しない場合は 0.0 を返す。
    """
    n = len(ranking)
    pos_indices = [i + 1 for i, item in enumerate(ranking) if item in relevant]
    num_pos = len(pos_indices)
    num_neg = n - num_pos
    if num_pos == 0 or num_neg == 0:
        return 0.0
    sum_ranks = sum(pos_indices)
    ideal_sum = num_pos * (num_pos + 1) / 2
    auc = (sum_ranks - ideal_sum) / (num_pos * num_neg)
    return auc


def coverage(all_recommended: List[List[Any]], all_items: Set[Any]) -> float:
    """Coverage を計算する関数。

    全ユーザーに対して推薦されたアイテムのユニーク数を、全アイテム数で割った値を返す。

    Args:
        all_recommended: 各ユーザー（またはクエリ）ごとの推薦アイテムリストのリスト。
        all_items: 全体のアイテム集合。

    Returns:
        Coverage の値（0.0～1.0）。
    """
    recommended_items: Set[Any] = set()
    for rec in all_recommended:
        recommended_items.update(rec)
    return len(recommended_items) / float(len(all_items)) if all_items else 0.0


def diversity_at_k(recommended: List[Any],
                   sim_func: Callable[[Any, Any], float],
                   k: int) -> float:
    """Diversity@k を計算する関数。

    推薦リストの上位 k 件内の各アイテムペアについて、
    類似度 sim_func(item1, item2) を 0～1 の値で返すと仮定し、(1 - 類似度) の平均値を多様性とする。

    Args:
        recommended: 推薦されたアイテムのリスト。
        sim_func: 2 つのアイテム間の類似度を計算する関数。返り値は 0～1 の実数。
        k: 評価対象とする上位件数。

    Returns:
        Diversity@k の値（0.0～1.0）。上位 k 件が 1 件未満の場合は 0.0 を返す。
    """
    rec_k = recommended[:k]
    if len(rec_k) < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(len(rec_k)):
        for j in range(i + 1, len(rec_k)):
            sim = sim_func(rec_k[i], rec_k[j])
            total += (1 - sim)
            count += 1
    return total / count if count > 0 else 0.0


def novelty_at_k(recommended: List[Any],
                 item_popularity: Dict[Any, float],
                 k: int) -> float:
    """Novelty@k を計算する関数。

    各推薦アイテムの情報量 -log2(probability) の平均値を計算する。
    item_popularity は各アイテムの出現確率または頻度（正の値）を返す辞書とする。

    Args:
        recommended: 推薦されたアイテムのリスト。
        item_popularity: アイテムごとの人気度（確率または頻度）の辞書。
        k: 評価対象とする上位件数。

    Returns:
        Novelty@k の値。関連アイテムが 0 件の場合は 0.0 を返す。
    """
    rec_k = recommended[:k]
    total = 0.0
    count = 0
    for item in rec_k:
        prob = item_popularity.get(item, None)
        if prob is None or prob <= 0:
            continue
        total += -log2(prob)
        count += 1
    return total / count if count > 0 else 0.0