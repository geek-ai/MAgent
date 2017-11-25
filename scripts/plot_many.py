"""plot curve from many log files"""

import sys
import matplotlib.pyplot as plt
import numpy as np

rec_filename = sys.argv[1]
plot_key = sys.argv[2]
list_col_index = int(sys.argv[3]) if len(sys.argv) > 3 else -1
silent = sys.argv[-1] == '--silent'

def parse_pair(item):
    """parse pair  \tkey: value\t """
    split_index = item.find(":")
    key = item[:split_index].strip()
    value = item[split_index+1:].strip()
    return key, value

def parse_log_file(filename, begin_item_index=0):
    """log_file format   \tkey: value\t key:value\t key:value\t ... """
    ret = {}
    with open(filename, 'r') as fin:
        for line in fin.readlines():
            items = line.split('\t')

            if len(items) < 1:  # ignore error
                continue

            for item in items[begin_item_index:]:
                key, value = parse_pair(item)
                if key not in ret:
                    ret[key] = []
                ret[key].append(value)

    return ret


rec_dict = parse_log_file(rec_filename)


legend = []
data = []
for log_file_name in rec_dict["log_file"]: # parse every file
    log_dict = parse_log_file(log_file_name)
    now = log_dict[plot_key]

    tmp = eval(now[0])
    if isinstance(tmp, list): # is list, expand it
        col_num = len(tmp)
        for row in range(len(now)):
            now[row] = eval(now[row])
        now = np.array(now)

        print(now)

        if list_col_index == -1:
            for col in range(col_num):
                legend.append(log_file_name + "-" + str(col))
                data.append(now[:,col])
        else:
            legend.append(log_file_name)
            data.append(now[:,list_col_index])
    else:  # is a scalar
        for i in range(len(now)):
            now[i] = eval(now[i])
        legend.append(log_file_name)
        data.append(now)

data = np.array(data)

print(legend)
print(data)
plt.plot(data.T)
plt.legend(legend)
plt.savefig(rec_filename + ".png")
if not silent:
    plt.show()
