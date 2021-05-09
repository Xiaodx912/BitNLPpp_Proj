import pandas as pd
import seaborn as sns
import json

from matplotlib import pyplot as plt

d1 = json.load(open('model/log_1620457009.json'))
d2 = json.load(open('model/log_1620467215.json'))
d = d1 + d2
da = pd.DataFrame(d)
l = []
for i in d:
    l.append({'epoch': i['epoch'], 'type': 'loss', 'data': i['loss']})
    if 'F1' in i:
        l.append({'epoch': i['epoch'], 'type': 'F1', 'data': i['F1']})

dl = pd.DataFrame(l)

plt.figure(figsize=(16, 10))
sns.set_palette('husl')
ax1 = sns.lineplot(x="epoch", y="data", hue='type', data=dl)
ax1.set_xlabel('Epochs')
ax1.set_ylabel('loss/F1')
ax1.set_ylim(0, 0.8)
h1 = plt.gca().get_lines()
ax2 = ax1.twinx()
ax2 = sns.lineplot(x="epoch", y="LR", data=da, linestyle='--', label='LR', legend=False)
ax2.set_ylabel('LearningRate')
ax2.set_ylim(0, 0.06)
h2 = plt.gca().get_lines()
h = h1 + h2
ax1.legend(handles=h[2:5], loc=1)

plt.savefig('fig.png')
