import json
import ast
import matplotlib.pyplot as plt
import numpy as np

# 从文件中读取数据
with open('winrate_data.json', 'r') as f:
    data = json.load(f)

data = {ast.literal_eval(key): value for key, value in data.items()}

# 按照胜率对数据进行排序，取前20个
sorted_data = sorted(data.items(), key=lambda x: x[1][0], reverse=True)
top_20_data = sorted_data[:20]

hands = [str(hand) for hand, value in top_20_data]
winrates = [value[0] for hand, value in top_20_data]

cmap = plt.get_cmap('Blues')  # 选择 'Blues' 渐变色

norm = plt.Normalize(0.53, 0.85)

colors = [cmap(norm(winrate)) for winrate in winrates]

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.barh(hands, winrates, color=colors)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
fig.colorbar(sm, ax=ax, label='Winrate')

ax.set_xlabel('Winrate')
ax.set_title('Top 20 Poker Hands Winrate')

ax.invert_yaxis()
plt.show()


for key, value in sorted_data:
    print(f"Key: {key}, Value: {value}")