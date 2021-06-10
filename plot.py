import pandas as pd
import seaborn as sns
import json

from matplotlib import pyplot as plt

path_list = ['model/log_1623302922.json']

d = []
for path in path_list:
    d += json.load(open(path))
da = pd.DataFrame(d)
l = []
for i in d:
    l.append({'epoch': i['e'], 'type': 'loss', 'data': i['loss']})
    if 'BO' in i:
        l.append({'epoch': i['e'], 'type': 'BO_F1', 'data': i['BO']})
        l.append({'epoch': i['e'], 'type': 'Macro_average_F1', 'data': i['Mac']})
        l.append({'epoch': i['e'], 'type': 'Micro_average_F1', 'data': i['Mic']})

dl = pd.DataFrame(l)

plt.figure(figsize=(16, 10))
sns.set_palette('husl')
ax1 = sns.lineplot(x="epoch", y="data", hue='type', data=dl)
ax1.set_xlabel('Epochs')
#ax1.set_xlim(0,20)
ax1.set_ylabel('loss/F1')
ax1.set_ylim(0, 1.1)
h1 = plt.gca().get_lines()
ax2 = ax1.twinx()
ax2 = sns.lineplot(x="e", y="LR", data=da, linestyle='--', label='LR', legend=False)
ax2.set_ylabel('LearningRate')
ax2.set_ylim(0, 0.06)
h2 = plt.gca().get_lines()
h = h1 + h2
ax1.legend(handles=h[4:], loc=1)

'''
ax1.set_xscale("log")
ax2.set_xscale("log")
'''
#plt.show()
plt.savefig('fig.png')
