import sys
from pyspark import SparkConf, SparkContext


def map_gen_pairs(lines):
    user, friends = lines.split('\t')
    friends = friends.split(',')

    # user has no friend, return empty list
    if len(friends) == 0:
        return []

    # create list of users who have common friends
    common_friend_list = []
    for i in range(len(friends) - 1):
        for j in range(i + 1, len(friends)):
            common_friend_list.append(((friends[i], friends[j]), 1))
            common_friend_list.append(((friends[j], friends[i]), 1))

    # create list of users that are already friends
    friend_list = []
    for i in range(len(friends)):
        friend_list.append(((user, friends[i]), 0))

    all_friend_list = friend_list + common_friend_list

    return all_friend_list


def map_find_recommendation(potential_friends):
    user = potential_friends[0]
    friends = potential_friends[1]
    # for i in range(len(friends)):
    #     if friends[i] != '':
    #        friends[i] = int(friends[i])
    friends = sorted(friends.items(), key=lambda x: (-x[1], int(x[0])))
    top_10_friends = friends[0:10]
    top_10_friends_id = [i[0] for i in top_10_friends]
    return user, top_10_friends_id


def main(input):
    conf = SparkConf()
    sc = SparkContext(conf=conf)
    file = sc.textFile(input[0])

    # create friend pairs and mutual friend pairs
    pair_map = file.flatMap(lambda x: map_gen_pairs(x))

    # filter out pairs that already friends
    pair_check_friends = pair_map.reduceByKey(lambda a, b: a * b)
    pair_friends = pair_check_friends.filter(lambda (k, prod): prod == 0)
    pair_not_friends = pair_map.subtractByKey(pair_friends)

    # perform reduce
    pairs_to_consider = pair_not_friends.reduceByKey(lambda a, b: a + b)

    # perform map again to find recommendations
    pairs_to_consider_map = pairs_to_consider.map(lambda ((u, f), num): (u, (f, num))).groupByKey().map(
        lambda (u, recommend): (u, dict((f, cnt) for f, cnt in recommend)))

    pairs_to_recommend = pairs_to_consider_map.map(lambda u_f: map_find_recommendation(u_f))
    pairs_to_recommend_format = pairs_to_recommend.map(lambda (a, b): "{}\t{}".format(a, ",".join(b)))
    pairs_to_recommend_format.saveAsTextFile(input[1])


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage:')
        print(" $ bin/spark-submit path/to/q1.py <path/to/data_file> <path/to/output_file>")
        sys.exit(0)

    main(sys.argv[1:3])