import random

from simulation import *
import csv

deck = generate_deck()

epoches = 400000

rank_values = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10,
    'J': 11, 'Q': 12, 'K': 13, 'A': 14
}
suit_values = {'♠': 1, '♣': 2, '♦': 3, '♥': 4}
data = []
data_count = [0 for i in range(11)]


# 随机生成
for epoc in tqdm(range(epoches), desc='Iterations', colour='#4a65c4'):
    random.shuffle(deck)
    hand = deck[:9]

    winner, value1, value2 = get_winner(hand[:2],hand[2:4], hand[4:])

    # 牌面和花色拆开
    ranks_in_hand = [card[:-1] for card in hand]
    suits_in_hand = [card[-1] for card in hand]

    hand_ranks = [rank_values[rank] for rank in ranks_in_hand]
    hand_suits = [suit_values[suit] for suit in suits_in_hand]

    # 录入
    data.append([])
    for i in range(9):
        data[-1].append(hand_suits[i])
        data[-1].append(hand_ranks[i])
    data[-1].append(winner)




# 打乱行顺序
random.shuffle(data)

header = [
    "hole1_Suit", "hole1_Rank", "hole2_Suit", "hole2_Rank",
    "oppo1_Suit", "oppo1_Rank", "oppo2_Suit", "oppo2_Rank",
    "com1_Suit", "com1_Rank", "com2_Suit", "com2_Rank", "com3_Suit", "com3_Rank", "com4_Suit", "com4_Rank", "com5_Suit", "com5_Rank",
    "winner"
]

# 写入 CSV 文件
output_file = "poker_winner_data.csv"
with open(output_file, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    # 写入标题
    writer.writerow(header)
    # 写入数据
    writer.writerows(data)

from collections import Counter

# 定义 CSV 文件路径
input_file = "poker_winner_data.csv"

# 读取数据并统计最后一列
hand_results = []
with open(input_file, mode="r", encoding="utf-8") as file:
    reader = csv.reader(file)
    next(reader)  # 跳过标题行
    for row in reader:
        hand_results.append(int(row[-1]))  # 将最后一列数据添加到 hand_results 列表中

# 使用 Counter 统计每个数字的出现次数
result_counts = Counter(hand_results)

# 打印统计结果
print("手牌结果统计：")
for result, count in sorted(result_counts.items()):  # 按结果排序
    print(f"结果 {result}: {count} 次")
