import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend
import matplotlib.pyplot as plt
import seaborn as sns

def _compute_apk(targets, predictions, k):

    if len(predictions) > k:
        predictions = predictions[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predictions):
        if p in targets and p not in predictions[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not list(targets):
        return 0.0

    return score / min(len(targets), k)


def _compute_precision_recall(targets, predictions, k):

    pred = predictions[:k]
    num_hit = len(set(pred).intersection(set(targets)))
    precision = float(num_hit) / len(pred)
    recall = float(num_hit) / len(targets)
    return precision, recall

def _compute_hitk(targets, predictions, k):

    if len(predictions) > k:
        predictions = predictions[:k]

    hits = [p for p in predictions if p in targets]

    return len(hits) / k

def _compute_ndcgk(targets, predictions, k):

    if len(predictions) > k:
        predictions = predictions[:k]

    dcg = 0.0
    idcg = 0.0

    for i, p in enumerate(predictions):
        if p in targets:
            dcg += 1.0 / np.log2(i + 2)

    for i in range(min(len(targets), k)):
        idcg += 1.0 / np.log2(i + 2)

    if not list(targets):
        return 0.0

    return dcg / idcg

def evaluate_ranking(model, test, train=None, k=10):
    """
    Compute Precision@k, Recall@k, Hit@k, NDCG@k scores and average precision (AP).
    One score is given for every user with interactions in the test
    set, representing the AP, Precision@k, Recall@k, Hit@k and NDCG@k of all their
    test items.

    Parameters
    ----------
    model: fitted instance of a recommender model
        The model to evaluate.
    test: :class:`spotlight.interactions.Interactions`
        Test interactions.
    train: :class:`spotlight.interactions.Interactions`, optional
        Train interactions. If supplied, rated items in
        interactions will be excluded.
    k: int or array of int,
        The maximum number of predicted items
    """

    test = test.tocsr()

    if train is not None:
        train = train.tocsr()

    if not isinstance(k, list):
        ks = [k]
    else:
        ks = k

    hitks = [list() for _ in range(len(ks))]
    ndcgks = [list() for _ in range(len(ks))]
    
    for user_id, row in enumerate(test):

        if not len(row.indices):
            continue

        predictions = -model.predict(user_id)
        predictions = predictions.argsort()

        if train is not None:
            rated = set(train[user_id].indices)
        else:
            rated = []

        predictions = [p for p in predictions if p not in rated]

        targets = row.indices

        for i, _k in enumerate(ks):
            hitk = _compute_hitk(targets, predictions, _k)
            hitks[i].append(hitk)
            ndcgk = _compute_ndcgk(targets, predictions, _k)
            ndcgks[i].append(ndcgk)

    hitks = [np.array(i) for i in hitks]
    ndcgks = [np.array(i) for i in ndcgks]

    if not isinstance(k, list):
        hitks = hitks[0]
        ndcgks = ndcgks[0]

    return np.mean(ndcgks), np.mean(hitks)