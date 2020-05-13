import numpy as np

filename = 'user-shows.txt'
f = open(filename, 'r')

# compute P and Q
p = np.zeros((9985, 9985))
q = np.zeros((563, 563))
r = np.zeros((9985, 563))
i = 0
users = [0] * 563
for user in f:
    ui = list(int(u) for u in user.strip().split())
    p[i][i] = np.sum(ui)
    users = np.add(users, ui)
    r[i] = ui
    i = i + 1

for num in range(len(users)):
    q[num][num] = users[num]

# compute matrix tau for shows
q_half = q
for i in range(len(users)):
    if q_half[i][i] != 0:
        q_half[i][i] = 1 / np.sqrt(q_half[i][i])
tau_1 = np.dot(r, q_half)
tau_2 = np.dot(tau_1, np.transpose(r))
tau_3 = np.dot(tau_2, r)
tau_show = np.dot(tau_3, q_half)
s_show = tau_show[:, 0:100]
s_alex = s_show[499]

# find recommendations for Alex
alex = {}
for i in range(len(s_alex)):
    alex[i] = s_alex[i]
recommend_alex = dict(sorted(alex.items(), key=lambda x: (-x[1], x[0]))[0:5])

showfile = 'shows.txt'
f_show = open(showfile, 'r')
showi = []
for show in f_show:
    showi.append(show)

print('Top 5 Recommendations for Alex: ')
for show, score in recommend_alex.items():
    print(showi[show], score)


# compute matrix tau for users
p_half = p
users_num = 9985
for i in range(users_num):
    if p_half[i][i] != 0:
        p_half[i][i] = 1 / np.sqrt(p_half[i][i])
tau_1 = np.dot(np.transpose(r), p_half)
tau_2 = np.dot(tau_1, r)
tau_3 = np.dot(tau_2, np.transpose(r))
tau_show = np.dot(tau_3, p_half)
s_show = np.transpose(tau_show)[:, 0:100]
s_alex = s_show[499]

# find recommendations for Alex
alex = {}
for i in range(len(s_alex)):
    alex[i] = s_alex[i]
recommend_alex = dict(sorted(alex.items(), key=lambda x: (-x[1], x[0]))[0:5])

print('Top 5 Recommendations for Alex: ')
for show, score in recommend_alex.items():
    print(showi[show], score)