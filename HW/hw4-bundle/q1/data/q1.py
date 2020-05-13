import numpy as np
import matplotlib.pyplot as plt
import time

feature_file = 'features.txt'
target_file = 'target.txt'
data_features = []
data_target = []
f_tgt = open(target_file, 'r')
f_feature = open(feature_file, 'r')

for f in f_feature:
    features = f.split(",")
    features = list(map(int, features))
    data_features.append(features)

for i in f_tgt:
    data_target.append(int(i))

C = 100

################ BGD #################
# parameters
w_bgd = np.zeros(122)
b_bgd = 0.0
fk_bgd = [C * 6414]
k1 = 0
eta_bgd = 0.0000003
epsilon_bgd = 0.25
delta_percent_cost_bgd = 1.0

t1 = time.clock()
while delta_percent_cost_bgd > epsilon_bgd:
    # update w and b
    grad_sum_w = np.zeros(122)
    grad_sum_b = 0.0
    for j in range(len(data_target)):
        x = np.array(data_features[j])
        y = np.array(data_target[j])
        xw = np.dot(x, w_bgd)
        val = y * (xw + b_bgd)
        if val >= 1:
            grad_sum_w += 0
            grad_sum_b += 0

        else:
            grad_sum_w -= y * x
            grad_sum_b -= y
    w_bgd = w_bgd - eta_bgd * (w_bgd + C * grad_sum_w)
    b_bgd = b_bgd - eta_bgd * C * grad_sum_b
    k1 = k1 + 1

    # calculate function value at current step
    summ = 0
    for j in range(len(data_target)):
        x = np.array(data_features[j])
        y = np.array(data_target[j])
        xw = np.dot(x, w_bgd)
        val = y * (xw + b_bgd)
        if val >= 1:
            summ += 0
        else:
            summ += 1 - val
    f_current = 0.5 * np.dot(w_bgd, w_bgd) + summ * C
    fk_bgd.append(f_current)

    # calculate delta
    delta_percent_cost_bgd = abs(fk_bgd[k1 - 1] - fk_bgd[k1]) * 100 / fk_bgd[k1 - 1]
    # print(delta_cost)
    # print(k1)
    # print(fk)
t2 = time.clock()
t_bgd = t2 - t1
print('Time for BGD: ', t_bgd)
print('Number of iterations for BGD: ', k1)


################ SGD ################# (1599 iters)
# parameters
w_sgd = np.zeros(122)
b_sgd = 0.0
fk_sgd = [C * 6414]
eta_sgd = 0.0001
epsilon_sgd = 0.001
i = 1
k2 = 0
delta_cost_sgd = 0
# delta_percent_cost_sgd = 1.0
# delta_cost_sgd = 0.5 * delta_cost_sgd + 0.5 * delta_percent_cost_bgd

# random shuffle
for j in range(len(data_target)):
    data_features[j].append(data_target[j])
np.random.shuffle(data_features)

t3 = time.clock()
while True:
    grad_sum_w = np.zeros(122)
    grad_sum_b = 0.0
    x = np.array(data_features[i][:-1])
    y = np.array(data_features[i][-1])
    xw = np.dot(x, w_sgd)
    val = y * (xw + b_sgd)
    if val >= 1:
        grad_sum_w += 0
        grad_sum_b += 0

    else:
        grad_sum_w -= y * x
        grad_sum_b -= y
    w_sgd = w_sgd - eta_sgd * (w_sgd + C * grad_sum_w)
    b_sgd = b_sgd - eta_sgd * C * grad_sum_b
    i = i % 6414 + 1
    k2 = k2 + 1

    # calculate function value at current step
    summ = 0
    for j in range(len(data_features)):
        x = np.array(data_features[j][:-1])
        y = np.array(data_features[j][-1])
        xw = np.dot(x, w_sgd)
        val = y * (xw + b_sgd)
        if val >= 1:
            summ += 0
        else:
            summ += 1 - val
    f_current = 0.5 * np.dot(w_sgd, w_sgd) + summ * C
    fk_sgd.append(f_current)

    # calculate delta
    delta_percent_cost_sgd = abs(fk_sgd[k2 - 1] - fk_sgd[k2]) * 100 / fk_sgd[k2 - 1]
    delta_cost_sgd = delta_cost_sgd * 0.5 + 0.5 * delta_percent_cost_sgd
    if delta_cost_sgd <= epsilon_sgd:
        break
    # print(fk_sgd)
    # print(k2)
    # print(i)

t4 = time.clock()
t_sgd = t4 - t3
print('Time for SGD: ', t_sgd)
print('Number of iterations for SGD: ', k2)


################ MiniBGD ################# (1061 iters)
# parameters
w_minibgd = np.zeros(122)
b_minibgd = 0.0
fk_minibgd = [C * 6414]
eta_minibgd = 0.00001
epsilon_minibgd = 0.01
batch_sz = 20
l = 0
k3 = 0
delta_cost_minibgd = 0

t5 = time.clock()
while True:
    start = l * batch_sz + 1
    end = min(6414, (l + 1) * batch_sz)
    mini_batch_data = data_features[start:end]
    grad_sum_w = np.zeros(122)
    grad_sum_b = 0.0
    for j in range(len(mini_batch_data)):
        x = np.array(mini_batch_data[j][:-1])
        y = np.array(mini_batch_data[j][-1])
        xw = np.dot(x, w_minibgd)
        val = y * (xw + b_minibgd)
        if val >= 1:
            grad_sum_w += 0
            grad_sum_b += 0
        else:
            grad_sum_w -= y * x
            grad_sum_b -= y
    w_minibgd = w_minibgd - eta_minibgd * (w_minibgd + C * grad_sum_w)
    b_minibgd = b_minibgd - eta_minibgd * C * grad_sum_b
    l = int((l + 1) % ((6414 + batch_sz - 1)/batch_sz))
    k3 = k3 + 1

    # calculate function value at current step
    summ = 0
    for j in range(len(data_features)):
        x = np.array(data_features[j][:-1])
        y = np.array(data_features[j][-1])
        xw = np.dot(x, w_minibgd)
        val = y * (xw + b_minibgd)
        if val >= 1:
            summ += 0
        else:
            summ += 1 - val
    f_current = 0.5 * np.dot(w_minibgd, w_minibgd) + summ * C
    fk_minibgd.append(f_current)

    # calculate delta
    delta_percent_cost_minibgd = abs(fk_minibgd[k3 - 1] - fk_minibgd[k3]) * 100 / fk_minibgd[k3 - 1]
    delta_cost_minibgd = delta_cost_minibgd * 0.5 + 0.5 * delta_percent_cost_minibgd
    if delta_cost_minibgd <= epsilon_minibgd:
        break
    # print(k3)

t6 = time.clock()
t_minibgd = t6 - t5
print('Time for Mini-Batch GD: ', t_minibgd)
print('Number of iterations for Mini-Batch GD: ', k3)

# plot cost function vs. iterations
x1 = range(k1 + 1)
x2 = range(k2 + 1)
x3 = range(k3 + 1)
p1 = plt.plot(x1, fk_bgd, 'r')
p2 = plt.plot(x2, fk_sgd, 'b')
p3 = plt.plot(x3, fk_minibgd, 'g')
plt.ylabel("Cost Function Value")
plt.xlabel("Iterations")
plt.title("Cost vs. Iterations")
plt.legend((p1[0], p2[0], p3[0]), ('BGD', 'SGD', 'Mini-Batch GD'))
plt.show()
