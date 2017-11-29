"""deprecated"""

import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import numpy as np
import sys

filename = sys.argv[1]

data = []

with open(filename) as fin:
    for i, row in enumerate(fin.readlines()):
        row = eval(row)
        data.append(row)
        #if i > max_n:
        #    break

move_ave = None
alpha = 0.5

n = len(data)
print(n)
for i, row in enumerate(data):
    row = np.array(row)
    row = row + 2
    row = np.where(row > 0, row, np.zeros_like(row))
    if move_ave is None:
        move_ave = row
    else:
        move_ave = alpha * move_ave + (1 - alpha) * row
    lin = np.arange(len(row))
    row = np.log(row + 1e-5)
    lin = np.log(lin + 1)
    plt.plot(lin, move_ave, color=hsv_to_rgb((0.33 - 0.33 * i / n,1,1)))

plt.show()
