import pandas as pd
import seaborn as sns
import json

from matplotlib import pyplot as plt

path_list = ['model/log_1621252262.json', 'model/log_1621254511.json', 'model/log_1621255699.json']

d = []
for path in path_list:
    d += json.load(open(path))
da = pd.DataFrame(d)
l = []
for i in d:
    l.append({'epoch': i['epoch'], 'type': 'loss', 'data': i['loss']})
    if 'BO_F1' in i:
        l.append({'epoch': i['epoch'], 'type': 'BO_F1', 'data': i['BO_F1']})
        l.append({'epoch': i['epoch'], 'type': 'Macro_average_F1', 'data': i['Macro_average_F1']})
        l.append({'epoch': i['epoch'], 'type': 'Micro_average_F1', 'data': i['Micro_average_F1']})

dl = pd.DataFrame(l)

plt.figure(figsize=(16, 10))
sns.set_palette('husl')
ax1 = sns.lineplot(x="epoch", y="data", hue='type', data=dl)
ax1.set_xlabel('Epochs')
ax1.set_ylabel('loss/F1')
ax1.set_ylim(0, 1.1)
h1 = plt.gca().get_lines()
ax2 = ax1.twinx()
ax2 = sns.lineplot(x="epoch", y="LR", data=da, linestyle='--', label='LR', legend=False)
ax2.set_ylabel('LearningRate')
ax2.set_ylim(0, 0.06)
h2 = plt.gca().get_lines()
h = h1 + h2
ax1.legend(handles=h[4:], loc=1)

plt.savefig('fig.png')
