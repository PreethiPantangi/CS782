import gzip
from collections import defaultdict
from datetime import datetime
from sklearn.model_selection import train_test_split

def parse(path):
    with gzip.open(path, 'rt', encoding='utf-8') as g:
        for l in g:
            l = l.replace('true', 'True').replace('false', 'False')
            yield eval(l)

def datapreprocessing(dataset):
    countU = defaultdict(lambda: 0)
    countP = defaultdict(lambda: 0)
    line = 0

    f = open('datapreprocessing/reviews_' + dataset + '.txt', 'w')
    for l in parse('datapreprocessing/reviews_' + dataset + '.json.gz'):
        line += 1
        f.write(" ".join([l['reviewerID'], l['asin'], str(l['overall']), str(l['unixReviewTime'])]) + ' \n')
        asin = l['asin']
        rev = l['reviewerID']
        time = l['unixReviewTime']
        countU[rev] += 1
        countP[asin] += 1
    f.close()

    usermap = dict()
    usernum = 0
    itemmap = dict()
    itemnum = 0
    User = dict()
    for l in parse('datapreprocessing/reviews_' + dataset + '.json.gz'):
        line += 1
        asin = l['asin']
        rev = l['reviewerID']
        time = l['unixReviewTime']
        if countU[rev] < 5 or countP[asin] < 5:
            continue

        if rev in usermap:
            userid = usermap[rev]
        else:
            usernum += 1
            userid = usernum
            usermap[rev] = userid
            User[userid] = []
        if asin in itemmap:
            itemid = itemmap[asin]
        else:
            itemnum += 1
            itemid = itemnum
            itemmap[asin] = itemid
        User[userid].append([time, itemid])
    # sort reviews in User according to time

    for userid in User.keys():
        User[userid].sort(key=lambda x: x[0])

    print (usernum, itemnum)

    interactions = []
    f = open('data/' + dataset + '/' + dataset + '.txt', 'w')
    
    for user in User.keys():
        for i in User[user]:
            f.write('%d %d\n' % (user, i[1]))
            interactions.append((user, i[1]))


    train_interactions, test_interactions = train_test_split(interactions, test_size=0.2, random_state=42)
    with open('data/' + dataset + '/train.txt', 'w') as train_file:
        for user, item in train_interactions:
            train_file.write('%d %d\n' % (user, item))

    with open('data/' + dataset + '/test.txt', 'w') as test_file:
        for user, item in test_interactions:
            test_file.write('%d %d\n' % (user, item))

    f.close()
