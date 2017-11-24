"""plot general log file according to given indexes"""

import sys
import matplotlib.pyplot as plt
import numpy as np

filename = sys.argv[1]

data = []

with open(filename, 'r') as fin:
    for line in fin.readlines():
        items = line.split('\t')

        row = []
        for item in items[1:]:
            t = eval(item.split(':')[1])
            if isinstance(t, list):
                for x in t:
                    row.append(x)
            else:
                row.append(t)
        if len(row) > 0:
            data.append(row)

data = np.array(data)


for index in sys.argv[2:]:
    index = int(index)
    plt.plot(data[:, index])
plt.show()
