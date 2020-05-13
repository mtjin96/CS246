import numpy as np
import matplotlib.pyplot as plt

filename = 'ratings.train.txt'
# initialize p and q
k = 20
lr = 0.02
lambd = 0.1
iter = 40
p = {}
q = {}

# initialize q, p
f = open(filename, 'r')
for i in f:
    pairs = i.split('\t')
    user, movie, rate = pairs[0], pairs[1], int(pairs[2])
    if movie not in q:
        q[movie] = np.sqrt(5.0 / k) * np.random.rand(k)
    if user not in p:
        p[user] = np.sqrt(5.0 / k) * np.random.rand(k)

# train data
errors = []
for i in range(iter):
    f.seek(0)
    for line in f:
        pairs = line.split('\t')
        user, movie, rate = pairs[0], pairs[1], int(pairs[2])
        # compute error
        qi = q[movie]
        pu = p[user]
        err = 2 * (rate - np.dot(qi, pu.reshape(k, 1)))
        # update qi and pu
        q[movie] = qi + lr * (err * pu - 2 * lambd * qi)
        p[user] = pu + lr * (err * qi - 2 * lambd * pu)

    # compute total error
    error = 0
    f.seek(0)
    for line in f:
        paris = line.split('\t')
        user, movie, rate = paris[0], paris[1], int(paris[2])
        pu = p[user]
        error = error + (rate - np.dot(q[movie], pu.reshape(k, 1))) ** 2
    for key, pu in p.items():
        error = error + lambd * (np.linalg.norm(pu) ** 2)
    for key, qi in q.items():
        error = error + lambd * (np.linalg.norm(qi) ** 2)
    errors.append(error)

f.close()

# plot error
x = range(0, iter)
plt.plot(x, errors)
plt.ylabel("Error")
plt.xlabel("Iterations")
plt.show()
