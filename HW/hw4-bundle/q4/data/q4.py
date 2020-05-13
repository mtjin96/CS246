import numpy as np
import math
import matplotlib.pyplot as plt


counts_file = 'counts.txt'
words_stream_file = 'words_stream.txt'
hash_param_file = 'hash_params.txt'

p_val = 123457
delta = (math.e) ** (-5)
epsilon = (math.e) * (10 ** (-4))
n_bucket = math.floor(math.e / epsilon)


# hash function
def hash_fun(a, b, p, n_buckets, x):
    y = x % p
    hash_val = (a * y + b) % p
    return hash_val % n_buckets


# read in hash parameters
f_para = open(hash_param_file, 'r')
hash_param = []
for line in f_para:
    params = line.strip().split('\t')
    hash_param.append(params)
    hash_param = [[int(x) for x in lst] for lst in hash_param]
# print(hash_param)


r = len(hash_param)
hash_vals = np.zeros((r, n_bucket))

# read in words stream and estimate the frequency
f_words = open(words_stream_file, 'r')
t = 0
for line in f_words:
    x_val = line.strip()
    for j in range(len(hash_param)):
        hash_val = hash_fun(hash_param[j][0], hash_param[j][1], p_val, n_bucket, int(x_val))
        hash_vals[j][hash_val] += 1
    t += 1

# calculate the error
f_counts = open(counts_file, 'r')
errors = []
fi_ts = []
for line in f_counts:
    id_freq = line.strip().split('\t')
    id = int(id_freq[0])
    freq = int(id_freq[1])
    f_tilda = 9999999999
    for j in range(len(hash_param)):
        hash_val = hash_fun(hash_param[j][0], hash_param[j][1], p_val, n_bucket, id)
        f_approx = hash_vals[j][hash_val]
        f_tilda = min(f_approx, f_tilda)
    err = (f_tilda - freq) / freq
    errors.append(err)
    fi_t = freq / t
    fi_ts.append(fi_t)

# plot
fig, ax = plt.subplots()
ax.plot(fi_ts, errors, 'o', markersize=1)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_title('Relative Error vs. Frequency')
ax.set_ylabel('Error (in log scale)')
ax.set_xlabel('Frequency (in log scale)')
plt.show()