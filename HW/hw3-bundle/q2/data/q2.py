import sys
from pyspark import SparkConf, SparkContext

n = 1000
iter = 40
beta = 0.8


def map_gen_links(lines):
    links = []
    node_i, node_j = lines.split('\t')
    links.append(((int(node_i), int(node_j)), 1))
    return links


def map_gen_links_transpose(lines):
    links = []
    node_i, node_j = lines.split('\t')
    links.append(((int(node_j), int(node_i)), 1))
    return links


def map_find_outdegree(links, outdegree):
    column_cnt = dict(outdegree)
    degrees = []
    # i = links[0] - 1
    # j = links[1] - 1
    i = links[0]
    j = links[1]
    degrees.append(((j, i), 1.0 / column_cnt[i]))
    return degrees


def calculate_rj(x, r):
    j = x[0]
    i_s = x[1]
    r_next = []
    summ = 0
    for k in i_s.keys():
        summ = summ + beta * r[k] * i_s[k]

    r_next.append((j, summ + (1 - beta) * 1.0 / n))
    return dict(r_next)


def matrix_vec_multi(pairs, vec):
    row, col, val = pairs[0][0] - 1, pairs[0][1] - 1, float(pairs[1])
    res = []
    res.append((row, val * vec[col]))
    return res


def main(input):
    conf = SparkConf()
    sc = SparkContext(conf=conf)
    fileName = input
    file = sc.textFile(fileName).distinct()

    # find links between nodes
    links = file.flatMap(lambda x: map_gen_links(x))

    ############# PageRank Algorithm #############
    # find outdegree for each node
    M_entries = links.map(lambda (pair, cnt): (pair[0], cnt))
    M_entries = M_entries.reduceByKey(lambda a, b: a + b)
    outdegree = []
    for i in M_entries.take(n):
        outdegree.append(i)

    M_entries = links.flatMap(lambda x: map_find_outdegree(x[0], outdegree))
    node_j = M_entries.map(lambda ((j, i), di): (j, (i, di)))
    node_j = node_j.groupByKey().map(lambda (j, outdegree): (j, dict((i, di) for i, di in outdegree)))

    # initialize r
    r = {}
    for k in range(1, n+1):
        r[k] = 1.0 / n
    # iterate through using the pagerank algorithm
    for i in range(iter):
        r_rdd = node_j.map(lambda x: calculate_rj(x, r))
        # update r
        for item in r_rdd.take(n):
            r[item.keys()[0]] = item.values()[0]

    # print top 5 and bottom 5
    r_top = sorted(r.items(), key=lambda kv: -kv[1])[0:5]
    print('Top 5 page rank: ', r_top)

    r_bottom = sorted(r.items(), key=lambda kv: kv[1])[0:5]
    print('Bottom 5 page rank: ', r_bottom)

    ############# HITS Algorithm #############
    # find links between nodes
    links = file.flatMap(lambda x: map_gen_links(x))
    links_transpose = file.flatMap(lambda x: map_gen_links_transpose(x))

    # initialize h
    h = {}
    for idx in range(n):
        h[idx] = 1.0
    # h = np.ones((n, 1))

    for i in range(iter):
        # calculate a
        a = links_transpose.flatMap(lambda x: matrix_vec_multi(x, h))
        a = a.reduceByKey(lambda a, b: a + b)

        # normalize a
        a_max = sorted(a.take(n), key=lambda x: -x[1])[0][1]
        a = a.map(lambda (idx, val): (idx, 1.0 * val / float(a_max)))

        # calculate and update h
        a_copy = {}
        for item in a.take(n):
            a_copy[item[0]] = item[1]

        h_update = links.flatMap(lambda x: matrix_vec_multi(x, a_copy))
        h_update = h_update.reduceByKey(lambda a, b: a + b)

        # normalize h_update
        h_update_max = sorted(h_update.take(n), key=lambda x: -x[1])[0][1]
        h_update = h_update.map(lambda (idx, val): (idx, 1.0 * val / h_update_max))

        for item in h_update.take(n):
            h[item[0]] = item[1]

    # print top 5 and bottom 5
    h_top = sorted(h.items(), key=lambda x: -x[1])[0:5]
    print('Top 5 hubbiness score: ', h_top)

    a_top = sorted(a_copy.items(), key=lambda x: -x[1])[0:5]
    print('Top 5 authority score: ', a_top)

    h_bottom = sorted(h.items(), key=lambda x: x[1])[0:5]
    print('Bottom 5 hubbiness score: ', h_bottom)

    a_bottom = sorted(a_copy.items(), key=lambda x: x[1])[0:5]
    print('Bottom 5 authority score: ', a_bottom)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage:')
        print(" $ bin/spark-submit path/to/q2.py <path/to/data_file> ")
        sys.exit(0)

    main(sys.argv[1])