# read in text file
filenm = 'browsing.txt'

# support threshold
s = 100

# count items frequency
def count_frequent(freq_items, s):
    freq_I = {}
    for i in freq_items:
        if freq_items[i] > s:
            freq_I[i] = freq_items[i]
    return freq_I


# find and print top 5 rules
def print_top_5(conf):
    conf = sorted(conf.items(), key=lambda x: (-x[1], x[0][0]))
    top_5 = conf[0:5]
    for i in top_5:
        message = str(i[0][0]) + '-->' + str(i[0][1]) + ': ' + str(i[1])
        print(message)



f = open(filenm, 'r')
I1 = {}
I2 = {}
I3 = {}

# count frequent individual items
for line in f:
    items = [i for i in line.split()]
    for i in items:
        if i in I1:
            I1[i] = I1[i] + 1
        else:
            I1[i] = 1

# filter the frequent ones
freq_I1 = count_frequent(I1, s)

# count frequent pairs
f.seek(0)
for line in f:
    items = [i for i in line.split()]
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[i] in freq_I1 and items[j] in freq_I1:
                items_new = []
                items_new.append(items[i])
                items_new.append(items[j])
                items_new.sort()
                k = (items_new[0], items_new[1])
                if k in I2:
                    I2[k] = I2[k] + 1
                else:
                    I2[k] = 1

# filter the frequent ones
freq_I2 = count_frequent(I2, s)


# compute confidence score and print top 5
conf = {}
# for k, v in freq_I2.items():
for k in freq_I2.keys():
    i1_key, i2_key = k[0], k[1]
    v = freq_I2[k]
    conf_12 = v / freq_I1[i1_key]
    conf_21 = v / freq_I1[i2_key]
    k1 = (i1_key, i2_key)
    k2 = (i2_key, i1_key)
    conf[k1] = conf_12
    conf[k2] = conf_21
print_top_5(conf)


#count frequent triples
f.seek(0)
for line in f:
    items = [i for i in line.split()]
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            for l in range(j + 1, len(items)):
                items_new = []
                items_new.append(items[i])
                items_new.append(items[j])
                items_new.append(items[l])
                items_new.sort()
                #print(items_new)
                p1 = (items_new[0], items_new[1])
                p2 = (items_new[1], items_new[2])
                p3 = (items_new[0], items_new[2])
                if (p1 in freq_I2) and (p2 in freq_I2) and (p3 in freq_I2):
                    k = (items_new[0], items_new[1], items_new[2])
                    if k in I3:
                        I3[k] = I3[k] + 1
                    else:
                        I3[k] = 1

# filter the frequent ones
freq_I3 = count_frequent(I3, s)
f.close()


# compute confidence score and print top 5
conf2 = {}

for k in freq_I3.keys():
    v = freq_I3[k]
    i1_key = (k[0], k[1])
    i2_key = (k[0], k[2])
    i3_key = (k[1], k[2])
    conf2_123 = v / freq_I2[i1_key]
    conf2_132 = v / freq_I2[i2_key]
    conf2_231 = v / freq_I2[i3_key]
    k1 = (i1_key, k[2])
    k2 = (i2_key, k[1])
    k3 = (i3_key, k[0])
    conf2[k1] = conf2_123
    conf2[k2] = conf2_132
    conf2[k3] = conf2_231
print_top_5(conf2)




