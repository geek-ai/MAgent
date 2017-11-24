"""plot a heatmap for tournament"""
import matplotlib.pyplot as plt
import numpy as np

def plot_heatmap(x, y, z):
    x, y = np.meshgrid(y, x)
    fig, ax = plt.subplots()
    im = ax.pcolormesh(x, y, z)
    fig.colorbar(im)

def smooth(data, alpha, beta=None):
    beta = beta or alpha
    for i in range(0, len(data)):
        for j in range(1, len(data[0])):
            data[i][j] = alpha * data[i][j-1] + (1-alpha) * data[i][j]

    for j in range(0, len(data[0])):
        for i in range(1, len(data)):
            data[i][j] = alpha * data[i-1][j] + (1-alpha) * data[i][j]

    return data


filename = "detail.log"

data = []
round2index = {}
ct = 0

with open(filename) as fin:
    for line in fin.readlines():
        item = line.split("\t")
        l = eval(item[0].split(":")[1])[2]
        r = eval(item[1].split(":")[1])[2]
        rate = eval(item[2].split(":")[1])
        num = eval(item[3].split(":")[1])

        for no in [l, r]:
            if no not in round2index:
                round2index[no] = ct
                ct += 1

        data.append([l, r, rate, num])

heat_data = [[0.5 for _ in range(ct)] for _ in range(ct)]

for line in data:
    l = round2index[line[0]]
    r = round2index[line[1]]
    rate = line[2]
    num = line[3]
    heat_data[l][r] = rate
    heat_data[r][l] = 1 - rate

heat_data = smooth(heat_data, 0.8)
heat_data = np.array(heat_data)
rounds = np.sort(np.array(round2index.keys()))

pick = 60
heat_data = heat_data[:pick,:pick]
rounds = rounds[:pick]

plot_heatmap(rounds, rounds, heat_data)

plt.show()
