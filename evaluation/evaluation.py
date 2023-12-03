import numpy as np
import random
import copy

# Martians: _compute_hitk takes target, predictions, and an integer k. 
# It computes the hit@k metric, i.e., the proportion of next correct predictions within the top k predictions.
def _compute_hitk(targets, predictions, k):

    if len(predictions) > k:
        predictions = predictions[:k]

    hits = [p for p in predictions if p in targets]

    return len(hits) / k
# Martians: _compute_ndcgk calculates the Normalized Discounted Cumulative Gain (NDCG) at k for the 
# list of target and prediction values by using logarithmic weighting. 
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

# Martians: Single evaluation function to calculate hit and ndcg values without bias.
def evaluate_metrics(model=None, dataset=None, args=None, algorithm='sasrec', isTest=None):
    
    if not isinstance(args['k'], list):
        ks = [args['k']]
    else:
        ks = args['k']

    hitks = [list() for _ in range(len(ks))]
    ndcgks = [list() for _ in range(len(ks))]

    if algorithm == 'caser':
        [train, test] = dataset
        test = test.tocsr()

        if train is not None:
            train = train.tocsr()

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

        if not isinstance(10, list):
            hitks = hitks[0]
            ndcgks = ndcgks[0]

        return np.mean(ndcgks), np.mean(hitks)
    

    elif algorithm == 'sasrec':
        hits = []
        ndcgs = []

        [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
        
        if usernum>10000:
            users = random.sample(range(1, usernum + 1), 10000)
        else:
            users = range(1, usernum + 1)
        for u in users:
            if isTest:
                if len(train[u]) < 1 or len(test[u]) < 1: continue
            else:
                if len(train[u]) < 1 or len(valid[u]) < 1: continue

            seq = np.zeros([args['maxlen']], dtype=np.int32)
            idx = args['maxlen'] - 1
            for i in reversed(train[u]):
                seq[idx] = i
                idx -= 1
                if idx == -1: break

            rated = set(train[u])
            rated.add(0)
            target = valid[u][0]

            item_idx = np.arange(itemnum+1)

            predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
            predictions = predictions[0]
            predictions = predictions.argsort()

            hit = _compute_hitk(target, predictions, 10)
            hits.append(hit)

            ndcg = _compute_hitk(target, predictions, 10)
            ndcgs.append(ndcg)

        return np.mean(ndcgs), np.mean(hits)