import numpy as np
import matplotlib.pyplot as plt
# parameters:
k = 20
lamb = 0.1
iters = 40
eta = 0.03

trainfile = "ratings.train.txt"
myfile = open(trainfile, 'r')

q = {}
p = {}

# first traverse, initialize all q and p
for line in myfile:
    items = line.strip().split("\t")
    q_row = int(items[0])
    p_row = int(items[1])
    if q_row in q:
        pass
    else:
        q[q_row] = np.random.rand(k) * np.sqrt(5.0/float(k))
    if p_row in p:
        pass
    else:
        p[p_row] = np.random.rand(k) * np.sqrt(5.0/float(k))


# start to train the data
error_record = []
for ii in range(iters):
    # in each iter, open file again
    readfile = open(trainfile, 'r')
    for readline in readfile:
        ratings = readline.strip().split("\t")
        q_idx = int(ratings[0])
        p_idx = int(ratings[1])
        rate = int(ratings[2])

        qi = q[q_idx]
        pu = p[p_idx]
        pu_T = pu.reshape(k, 1)

        varep = 2.0 * (rate - np.dot(qi, pu_T))
        # update
        qi_new = qi + eta * (varep * pu - 2.0 * lamb * qi)
        pu_new = pu + eta * (varep * qi - 2.0 * lamb * pu)
        q[q_idx] = qi_new
        p[p_idx] = pu_new
    # calculate error
    error = 0.0
    readfile = open(trainfile, 'r')
    for readline in readfile:
        ratings = readline.strip().split("\t")
        q_idx = int(ratings[0])
        p_idx = int(ratings[1])
        rate = int(ratings[2])

        qi = q[q_idx]
        pu = p[p_idx]
        pu_T = pu.reshape(k, 1)
        error += (rate - np.dot(qi, pu_T)) ** 2
    for q_key in q:
        error += lamb * np.sum(q[q_key] * q[q_key])
    for p_key in p:
        error += lamb * np.sum(p[p_key] * p[p_key])
    # record error in each iter
    error_scalar = error.reshape(())
    print(str(ii + 1) + " of " + str(iters) + " iters: " + str(error_scalar))
    error_record.append(error_scalar)

x = np.arange(0, iters, 1) + 1
y = error_record
plt.plot(x, y, "-o")
plt.xlabel("# of Iteration")
plt.ylabel("Error")
plt.title("Error vs Iteration")
plt.show()