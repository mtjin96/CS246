import sys
from pyspark import SparkConf, SparkContext
from numpy import array
import numpy as np
import matplotlib.pyplot as plt


def find_centroids(parsed_data, centroids):
    distance = []
    for c in centroids:
        distance.append(np.linalg.norm((parsed_data - c)))
    c_idx = np.argmin(distance)
    #c = centroids[c_idx]
    return c_idx, parsed_data

def find_centroids_manhattan(parsed_data, centroids):
    distance = []
    for c in centroids:
        distance.append(np.linalg.norm((parsed_data - c), 1))
    c_idx = np.argmin(distance)
    #c = centroids[c_idx]
    return c_idx, parsed_data


def compute_cost(paris, centroids):
    idx, point = paris[0], paris[1]
    c = centroids[idx]
    distance = (np.linalg.norm((point - c))) ** 2
    # print('distance: ', distance)
    return 1, distance

def compute_cost_manhattan(paris, centroids):
    idx, point = paris[0], paris[1]
    c = centroids[idx]
    distance = np.linalg.norm((point - c), 1)
    # print('distance: ', distance)
    return 1, distance


def get_new_centroids(pairs):
    old, new = pairs[0], pairs[1]
    return new


def get_total_cost(cost):
    idx, costs = cost[0], cost[1]
    return costs


def main():
    conf = SparkConf()
    sc = SparkContext(conf=conf)
    fileName = sys.argv[1]
    file = sc.textFile(fileName)
    parsed_data = file.map(lambda l: array([float(x) for x in l.split(' ')]))
    c1 = np.loadtxt(sys.argv[2])
    c2 = np.loadtxt(sys.argv[3])
    max_iter = 20

    ###################################
    ####### Euclidean Distance ########
    ###################################
    # using c1 as initialized centroids
    costs = []
    for i in range(max_iter):
        # find closet centroid and assign the point to it
        c1_points = parsed_data.map(lambda x: find_centroids(x, c1))

        # compute cost
        cost = c1_points.map(lambda x: compute_cost(x, c1))
        cost = cost.groupByKey().mapValues(lambda x: sum(x))
        cost = cost.map(lambda x: get_total_cost(x))
        costs.append(cost.take(1))

        # get new centroids of groups
        c1_new_centroids = c1_points.groupByKey().mapValues(lambda x: sum(x) / len(x))
        c1_new_centroids = c1_new_centroids.map(lambda x: get_new_centroids(x))
        c1 = c1_new_centroids.take(10)

    plt.scatter(range(1, max_iter + 1), costs)
    plt.show()


    # using c2 as initialized centroids
    costs2 = []
    for i in range(max_iter):
        # find closet centroid and assign the point to it
        c2_points = parsed_data.map(lambda x: find_centroids(x, c2))

        # compute cost
        cost2 = c2_points.map(lambda x: compute_cost(x, c2))
        cost2 = cost2.groupByKey().mapValues(lambda x: sum(x))
        cost2 = cost2.map(lambda x: get_total_cost(x))
        costs2.append(cost2.take(1))

        # get new centroids of groups
        c2_new_centroids = c2_points.groupByKey().mapValues(lambda x: sum(x) / len(x))
        c2_new_centroids = c2_new_centroids.map(lambda x: get_new_centroids(x))
        c2 = c2_new_centroids.take(10)

    plt.scatter(range(1, max_iter + 1), costs2)
    plt.show()


    ###################################
    ####### Manhattan Distance ########
    ###################################
    c1 = np.loadtxt(sys.argv[2])
    c2 = np.loadtxt(sys.argv[3])
    # using c1 as initialized centroids
    costs_m1 = []
    for i in range(max_iter):
        # find closet centroid and assign the point to it
        c1_points = parsed_data.map(lambda x: find_centroids_manhattan(x, c1))

        # compute cost
        cost_m1 = c1_points.map(lambda x: compute_cost_manhattan(x, c1))
        cost_m1 = cost_m1.groupByKey().mapValues(lambda x: sum(x))
        cost_m1 = cost_m1.map(lambda x: get_total_cost(x))
        costs_m1.append(cost_m1.take(1))

        # get new centroids of groups
        c1_new_centroids = c1_points.groupByKey().mapValues(lambda x: sum(x) / len(x))
        c1_new_centroids = c1_new_centroids.map(lambda x: get_new_centroids(x))
        c1 = c1_new_centroids.take(10)

    plt.scatter(range(1, max_iter + 1), costs_m1)
    plt.show()


    # using c2 as initialized centroids
    costs_m2 = []
    for i in range(max_iter):
        # find closet centroid and assign the point to it
        c2_points = parsed_data.map(lambda x: find_centroids_manhattan(x, c2))

        # compute cost
        cost_m2 = c2_points.map(lambda x: compute_cost_manhattan(x, c2))
        cost_m2 = cost_m2.groupByKey().mapValues(lambda x: sum(x))
        cost_m2 = cost_m2.map(lambda x: get_total_cost(x))
        costs_m2.append(cost_m2.take(1))

        # get new centroids of groups
        c2_new_centroids = c2_points.groupByKey().mapValues(lambda x: sum(x) / len(x))
        c2_new_centroids = c2_new_centroids.map(lambda x: get_new_centroids(x))
        c2 = c2_new_centroids.take(10)

    plt.scatter(range(1, max_iter + 1), costs_m2)
    plt.show()
    #print('cost', costs_m2)
    print('cost_eud_1', costs)
    print('cost_edu_2', costs2)
    print('cost_man_1', costs_m1)

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Usage:')
        print(" $ bin/spark-submit path/to/q1.py <path/to/data_file> <path/to/c1_file> <path/to/c2_file>")
        sys.exit(0)

    main()