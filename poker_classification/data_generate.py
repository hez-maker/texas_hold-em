import random

from simulation import *
import csv

deck = generate_deck()

epoches = 2000000
y_num = 5000
rank_values = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10,
    'J': 11, 'Q': 12, 'K': 13, 'A': 14
}
suit_values = {'♠': 1, '♣': 2, '♦': 3, '♥': 4}
data = []
data_count = [0 for i in range(11)]
y_num_list = [y_num, y_num, y_num-1200, y_num, y_num-1200, y_num-1400, y_num-2000, y_num-1400, y_num-1000, y_num-1500]

# 随机生成
for epoc in tqdm(range(epoches), desc='Iterations', colour='#4a65c4'):
    random.shuffle(deck)
    hand = deck[:7]

    score, hand_value = evaluate(hand, [])
    if data_count[score] < y_num_list[score]:
        data_count[score] += 1
        # 牌面和花色拆开
        ranks_in_hand = [card[:-1] for card in hand]
        suits_in_hand = [card[-1] for card in hand]

        hand_ranks = [rank_values[rank] for rank in ranks_in_hand]
        hand_suits = [suit_values[suit] for suit in suits_in_hand]

        # 录入
        data.append([])
        for i in range(7):
            data[-1].append(hand_suits[i])
            data[-1].append(hand_ranks[i])
        data[-1].append(score)


def data_write(hand):
    score, hand_value = evaluate(hand, [])

    data_count[score] += 1

    # 牌面和花色拆开
    ranks_in_hand = [card[:-1] for card in hand]
    suits_in_hand = [card[-1] for card in hand]

    hand_ranks = [rank_values[rank] for rank in ranks_in_hand]
    hand_suits = [suit_values[suit] for suit in suits_in_hand]

    # 录入
    data.append([])
    for i in range(7):
        data[-1].append(hand_suits[i])
        data[-1].append(hand_ranks[i])
    data[-1].append(score)


# 加入1200个两对2221样本
for i in range(1200):
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
    suits = ['♠', '♣', '♦', '♥']
    rank1 = random.sample(ranks, 3)
    suit1 = [random.sample(suits, 2) for _ in rank1]
    hand = [f'{rank}{suit}' for rank, suit_list in zip(rank1, suit1) for suit in suit_list]
    left_deck = [card for card in deck if card not in hand]
    random.shuffle(left_deck)
    hand += left_deck[:1]
    random.shuffle(hand)

    data_write(hand)

# 加入1200个顺子A2345样本
for i in range(1200):
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
    suits = ['♠', '♣', '♦', '♥']
    rank1 = ['A', '2', '3', '4', '5']
    suit1 = [random.sample(suits, 1) for _ in rank1]
    hand = [f'{rank}{suit}' for rank, suit_list in zip(rank1, suit1) for suit in suit_list]
    left_deck = [card for card in deck if card not in hand]
    random.shuffle(left_deck)
    hand += left_deck[:2]
    random.shuffle(hand)

    data_write(hand)

# 加入700个6同花
for i in range(700):
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
    suits = ['♠', '♣', '♦', '♥']
    rank1 = random.sample(ranks, 6)
    suit1 = random.sample(suits, 1)
    hand = [f'{rank}{suit1[0]}' for rank in rank1]
    left_deck = [card for card in deck if card not in hand]
    random.shuffle(left_deck)
    hand += left_deck[:1]
    random.shuffle(hand)

    data_write(hand)

# 加入700个7同花
for i in range(700):
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
    suits = ['♠', '♣', '♦', '♥']
    rank1 = random.sample(ranks, 7)
    suit1 = random.sample(suits, 1)
    hand = [f'{rank}{suit1[0]}' for rank in rank1]
    random.shuffle(hand)

    data_write(hand)

# 加入1000个葫芦331样本
for i in range(1000):
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
    suits = ['♠', '♣', '♦', '♥']
    rank1 = random.sample(ranks, 2)
    suit1 = [random.sample(suits, 3) for _ in rank1]
    hand = [f'{rank}{suit}' for rank, suit_list in zip(rank1, suit1) for suit in suit_list]
    left_deck = [card for card in deck if card not in hand]
    random.shuffle(left_deck)
    hand += left_deck[:1]
    random.shuffle(hand)

    data_write(hand)

# 加入1000个葫芦322样本
for i in range(1000):
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
    suits = ['♠', '♣', '♦', '♥']
    rank1 = random.sample(ranks, 3)
    suit1 = random.sample(suits, 3)
    hand = [f'{rank1[0]}{suit}' for suit in suit1]
    suit2 = random.sample(suits, 2)
    hand += [f'{rank1[1]}{suit}' for suit in suit2]
    suit3 = random.sample(suits, 2)
    hand += [f'{rank1[2]}{suit}' for suit in suit3]
    random.shuffle(hand)

    data_write(hand)

# 加入四条样本
for i in range(3600 - data_count[7]):
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
    suits = ['♠', '♣', '♦', '♥']
    rank = random.choice(ranks)
    hand = [f'{rank}{suit}' for suit in suits]
    left_deck = [card for card in deck if card not in hand]
    random.shuffle(left_deck)
    hand += left_deck[:3]
    random.shuffle(hand)

    data_write(hand)

# 加入700个四条43样本
for i in range(700):
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
    suits = ['♠', '♣', '♦', '♥']
    rank1 = random.sample(ranks, 2)
    hand = [f'{rank1[0]}{suit}' for suit in suits]

    suit2 = random.sample(suits, 3)
    hand += [f'{rank1[1]}{suit}' for suit in suit2]

    random.shuffle(hand)

    data_write(hand)

# 加入700个四条421样本
for i in range(700):
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
    suits = ['♠', '♣', '♦', '♥']
    rank1 = random.sample(ranks, 3)
    hand = [f'{rank1[0]}{suit}' for suit in suits]

    suit2 = random.sample(suits, 2)
    hand += [f'{rank1[1]}{suit}' for suit in suit2]

    suit3 = random.sample(suits, 1)
    hand += [f'{rank1[2]}{suit}' for suit in suit3]

    random.shuffle(hand)

    data_write(hand)

# 加入同花顺样本
for i in range(4000 - data_count[8]):
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
    suits = ['♠', '♣', '♦', '♥']
    suit = random.choice(suits)
    index = random.randint(0, 7)
    hand = [f'{ranks[index + i]}{suit}' for i in range(5)]
    left_deck = [card for card in deck if card not in hand]
    random.shuffle(left_deck)
    hand += left_deck[:2]
    random.shuffle(hand)

    data_write(hand)

# 加入1000个同花顺A2345样本
for i in range(1000):
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
    suits = ['♠', '♣', '♦', '♥']
    rank1 = ['A', '2', '3', '4', '5']
    suit1 = random.sample(suits, 1)
    hand = [f'{rank}{suit1[0]}' for rank in rank1]

    left_deck = [card for card in deck if card not in hand]
    random.shuffle(left_deck)
    hand += left_deck[:2]
    random.shuffle(hand)

    data_write(hand)

# 加入皇家同花顺样本
for i in range(y_num_list[9] - data_count[9]):
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
    suits = ['♠', '♣', '♦', '♥']
    suit = random.choice(suits)
    index = 8
    hand = [f'{ranks[index + i]}{suit}' for i in range(5)]
    left_deck = [card for card in deck if card not in hand]
    random.shuffle(left_deck)
    hand += left_deck[:2]
    random.shuffle(hand)

    data_write(hand)

# 打乱行顺序
random.shuffle(data)

header = [
    "Card1_Suit", "Card1_Rank", "Card2_Suit", "Card2_Rank", "Card3_Suit", "Card3_Rank",
    "Card4_Suit", "Card4_Rank", "Card5_Suit", "Card5_Rank", "Card6_Suit", "Card6_Rank", "Card7_Suit", "Card7_Rank",
    "Hand_Result"
]

# 写入 CSV 文件
output_file = "poker_hand_data2.csv"
with open(output_file, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    # 写入标题
    writer.writerow(header)
    # 写入数据
    writer.writerows(data)

from collections import Counter

# 定义 CSV 文件路径
input_file = output_file

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
