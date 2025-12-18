# 蒙特卡洛模拟
import copy
import random
from collections import Counter
from tqdm import tqdm


def longest_consecutive_subsequence(nums):
    """
    查找最长连续递减子串
    :param nums: 从大到小排序后列表
    :return: 最长连续个数，最长字串
    """

    longest_streak = 1
    current_streak = 1
    index = 0

    # 遍历排序后的数组，查找连续的子串
    for i in range(1, len(nums)):
        if nums[i] == nums[i - 1] - 1:
            current_streak += 1
        else:
            if longest_streak < current_streak:
                longest_streak = current_streak
                index = i
            current_streak = 1  # Reset current streak

    if longest_streak < current_streak:
        longest_streak = current_streak
        index = i + 1

    return longest_streak, nums[index - longest_streak:index]


def evaluate(hole_cards, community_cards):
    """
    7张牌判断牌型
    :param hand: list, eg. ['3♣', '6♠', '7♥', '10♦', 'J♠', '4♣', '7♠',]
    :return: tuple, (牌型, 牌面值 list[])
    """

    cards = hole_cards + community_cards  # 7张牌
    rank_values = {
        '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10,
        'J': 11, 'Q': 12, 'K': 13, 'A': 14
    }
    suit_values = {
        '♠': 1, '♣': 2, '♦': 3, '♥': 4
    }

    # 编码，按扑克值从大到小排序
    coded_cards = sorted([(rank_values[card[:-1]], suit_values[card[-1]]) for card in cards], key=lambda x: x[0],
                         reverse=True)

    # 牌面和花色拆开
    ranks_in_hand = [card[:-1] for card in cards]
    suits_in_hand = [card[-1] for card in cards]

    # 统计牌面值和花色的频次
    rank_counts = Counter(ranks_in_hand)  # {'3': 3, '5': 2, '2': 2}
    suit_counts = Counter(suits_in_hand)  # {'♠': 4, '♥': 1, '♦': 3}

    # 找同花顺
    suit_rank = {1: [], 2: [], 3: [], 4: []}  # 记录花色下点数
    for i in range(7):
        suit_rank[coded_cards[i][1]].append(coded_cards[i][0])
    for i in range(7):  # 加入1
        if coded_cards[i][0] == 14:
            suit_rank[coded_cards[i][1]].append(1)
        else:
            break
    for key, value in suit_rank.items():
        longest_streak, longest_list = longest_consecutive_subsequence(value)
        if longest_streak >= 5:
            return (9 if longest_list[0] == 14 else 8, longest_list[:5])

    # 找4条
    if 4 in rank_counts.values():
        four = [rank_values[key] for key, value in rank_counts.items() if value == 4]
        kicker = []
        for i, num in enumerate(coded_cards):
            if num[0] != four[0]:
                kicker.append(num[0])
                break
        return (7, four + kicker)

    # 找葫芦
    three = [rank_values[key] for key, value in rank_counts.items() if value == 3]
    if len(three) == 2:
        three = sorted(three, reverse=True)
        return (6, three)
    elif 3 in rank_counts.values() and 2 in rank_counts.values():
        two = [rank_values[key] for key, value in rank_counts.items() if value == 2]
        two = sorted(two, reverse=True)
        return (6, [three[0], two[0]])

    # 找同花
    if any(count >= 5 for count in suit_counts.values()):
        suit = suit_values[[key for key, value in suit_counts.items() if value >= 5][0]]
        return (5, [card[0] for card in coded_cards if card[1] == suit][:5])

    # 找顺子
    nums = [rank_values[key] for key, value in rank_counts.items()]
    nums = sorted(nums, reverse=True)
    if 'A' in rank_counts:
        nums.append(1)
    straight_num, straight_card = longest_consecutive_subsequence(nums)
    if straight_num >= 5:
        return (4, straight_card[:5])

    # 找三条
    if 3 in rank_counts.values():
        three = [rank_values[key] for key, value in rank_counts.items() if value == 3]
        kicker = []
        for i, num in enumerate(coded_cards):
            if num[0] != three[0]:
                kicker.append(num[0])
        return (3, three + kicker[:2])

    # 找两对
    two = [rank_values[key] for key, value in rank_counts.items() if value == 2]
    two = sorted(two, reverse=True)
    if len(two) >= 2:
        two = two[:2]
        kicker = []
        for i, num in enumerate(coded_cards):
            if num[0] != two[0] and num[0] != two[1]:
                kicker.append(num[0])
                break
        return (2, two + kicker)
    elif len(two) == 1:
        kicker = []
        for i, num in enumerate(coded_cards):
            if num[0] != two[0]:
                kicker.append(num[0])
        return (1, two + kicker[:3])

    # 高牌
    return (0, [rank for rank, suit in coded_cards][:5])


def get_winner(hole_cards1, hole_cards2, community_cards):
    score1, hand_value1 = evaluate(hole_cards1, community_cards)
    score2, hand_value2 = evaluate(hole_cards2, community_cards)

    if score1 > score2:
        return 1, (score1, hand_value1), (score2, hand_value2)
    elif score1 < score2:
        return 2, (score1, hand_value1), (score2, hand_value2)
    else:
        for v1, v2 in zip(hand_value1, hand_value2):
            if v1 > v2:
                return 1, (score1, hand_value1), (score2, hand_value2)
            elif v1 < v2:
                return 2, (score1, hand_value1), (score2, hand_value2)
        return 0, (score1, hand_value1), (score2, hand_value2)


def generate_deck():
    # 定义花色和牌面
    suits = ['♠', '♣', '♦', '♥']
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
    # 创建一副扑克牌
    deck = [f'{rank}{suit}' for rank in ranks for suit in suits]
    random.shuffle(deck)
    return deck


class simulation:
    def __init__(self, hole_cards, opp_cards, community_cards, iterations=1000):
        self.hole_cards = hole_cards
        self.opp_cards = opp_cards
        self.community_cards = community_cards
        self.iterations = iterations
        self.deck_left = generate_deck()
        self.deck_left = [card for card in self.deck_left if card not in hole_cards + community_cards + opp_cards]
        self.win_count = 0
        self.dogfall = 0
        self.cards_value = [0 for _ in range(10)]

    def run(self):
        for iter in tqdm(range(self.iterations), desc='Iterations', colour='#4a65c4'):
            # 补全手牌
            random.shuffle(self.deck_left)  # 洗牌
            community_cards = self.community_cards.copy()
            opp_cards = self.opp_cards.copy()
            if len(self.opp_cards) == 0:
                opp_cards = self.deck_left[:2]
                community_cards += self.deck_left[2:7 - len(self.community_cards)]
            else:
                community_cards += self.deck_left[:5 - len(self.community_cards)]
            winner, value1, value2 = get_winner(self.hole_cards, opp_cards, community_cards)
            self.cards_value[value1[0]] += 1
            if winner == 1:
                self.win_count += 1
            if winner == 0:
                self.dogfall += 1


if __name__ == '__main__':
    # 牌面
    s = ['♠', '♣', '♦', '♥']
    c = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']

    # iterations = 5000000
    # data = {}
    # for index, i in enumerate(c):
    #     for j in c[index:]:
    #         # 异色
    #         hole_cards = [f'{i}{s[0]}', f'{j}{s[1]}']
    #         opp_cards = []
    #         community_cards = []
    #         sim = simulation(hole_cards, opp_cards, community_cards, iterations)
    #         sim.run()
    #         data[(j, i, 2)] = [sim.win_count / sim.iterations, sim.cards_value]
    #         # 同色
    #         if i == j:
    #             continue
    #         hole_cards = [f'{i}{s[0]}', f'{j}{s[0]}']
    #         opp_cards = []
    #         community_cards = []
    #         sim = simulation(hole_cards, opp_cards, community_cards, iterations)
    #         sim.run()
    #         data[(j, i, 1)] = [sim.win_count / sim.iterations, sim.cards_value]
    #
    # data = {str(key): value for key, value in data.items()}
    # print(data)

    import json

    # 将字典保存为 JSON 格式的文件
    # with open("winrate_data.json", "w") as f:
    #     json.dump(data, f)

    hole_cards = ['A♦', 'K♦']
    opp_cards = ['2♠', '2♥']
    community_cards = []
    sim = simulation(hole_cards, opp_cards, community_cards, 100000)
    sim.run()
    print(sim.win_count / sim.iterations, sim.dogfall / sim.iterations)
    print(sim.cards_value)
