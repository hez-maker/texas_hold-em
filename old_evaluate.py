import random
from collections import Counter
import itertools
from tqdm import tqdm
import copy


def generate_deck():
    # 定义花色和牌面
    suits = ['♠', '♣', '♦', '♥']
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
    # 创建一副扑克牌
    deck = [f'{rank}{suit}' for rank in ranks for suit in suits]
    return shuffle_deck(deck)


# 洗牌函数
def shuffle_deck(deck):
    deck_copy = deck.copy()
    random.shuffle(deck_copy)
    return deck_copy


# 发牌函数：给玩家和公共牌发牌
def deal_hands(deck):
    deck = shuffle_deck(deck)
    player1_hand = deck[:2]  # 玩家1手牌
    player2_hand = deck[2:4]  # 玩家2手牌
    community_cards = deck[4:9]  # 公共牌（包括flop、turn、river）
    return player1_hand, player2_hand, community_cards


def evaluate_hand(hand):
    """
    5张牌判断牌型
    :param hand: list, eg. ['3♣', '6♠', '7♥', '10♦', 'J♠']
    :return: tuple, (牌型, 牌面值 list[])
    """

    rank_values = {
        '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10,
        'J': 11, 'Q': 12, 'K': 13, 'A': 14
    }
    suit_values = {
        '♠': 1, '♣': 2, '♦': 3, '♥': 4
    }

    # 牌面和花色拆开，注意是str
    ranks_in_hand = [card[:-1] for card in hand]
    suits_in_hand = [card[-1] for card in hand]

    # 编码，按扑克值从大到小排序
    cards = sorted([(rank_values[card[:-1]], suit_values[card[-1]]) for card in hand], key=lambda x: x[0], reverse=True)

    # 统计牌面值和花色的频次
    rank_counts = Counter(ranks_in_hand)
    suit_counts = Counter(suits_in_hand)

    # 获取所有的牌面值（转化为int）
    hand_values = sorted([rank_values[rank] for rank in ranks_in_hand], reverse=True)

    # 判断是否为同花
    is_flush = len(suit_counts) == 1

    # 判断是否为顺子
    if set(hand_values) == {2, 3, 4, 5, 14}:  # 处理 A 作为 1 的情况
        hand_values = [5, 4, 3, 2, 1]
    is_straight = len(rank_counts) == 5 and (hand_values[0] - hand_values[4] == 4)

    # 判断牌型
    if is_flush and is_straight:
        if hand_values == [14, 13, 12, 11, 10]:  # 皇家同花顺
            return (9, hand_values)  # 返回 (优先级, 牌面值)
        return (8, hand_values)  # 同花顺
    elif 4 in rank_counts.values():  # 四条
        four = [key for key, value in rank_counts.items() if value == 4][0]
        kicker = [key for key, value in rank_counts.items() if value == 1][0]
        return (7, [rank_values[four], rank_values[kicker]])
    elif 3 in rank_counts.values() and 2 in rank_counts.values():  # 葫芦
        three = [key for key, value in rank_counts.items() if value == 3][0]
        two = [key for key, value in rank_counts.items() if value == 2][0]
        return (6, [rank_values[three], rank_values[two]])
    elif is_flush:  # 同花
        return (5, hand_values)
    elif is_straight:  # 顺子
        return (4, hand_values)
    elif 3 in rank_counts.values():  # 三条
        three = [key for key, value in rank_counts.items() if value == 3][0]
        kickers = sorted([rank_values[key] for key, value in rank_counts.items() if value == 1], reverse=True)
        return (3, [rank_values[three]] + kickers)
    elif list(rank_counts.values()).count(2) == 2:  # 两对
        pairs = sorted([rank_values[key] for key, value in rank_counts.items() if value == 2], reverse=True)
        kicker = [key for key, value in rank_counts.items() if value == 1][0]
        return (2, pairs+[rank_values[kicker]])
    elif 2 in rank_counts.values():  # 对子
        pair = [key for key, value in rank_counts.items() if value == 2][0]
        kickers = sorted([rank_values[key] for key, value in rank_counts.items() if value == 1], reverse=True)
        return (1, [rank_values[pair]] + kickers)
    else:  # 高牌
        return (0, hand_values)


def compare_hands(score1, hand_values1, score2, hand_values2):
    """
    比较 5张牌牌力
    :param score1: 牌 1分数
    :param hand_values1: 牌 1牌面
    :param score2: 牌 2分数
    :param hand_values2: 牌 2牌面
    :return: 0:平局, 1:1赢, 2:2赢
    """

    if score1 > score2:
        return 1
    elif score1 < score2:
        return 2
    else:
        for v1, v2 in zip(hand_values1, hand_values2):
            if v1 > v2:
                return 1
            elif v1 < v2:
                return 2
        return 0


def best_hand(hole_cards, community_cards):
    """
    从手牌+公共牌中找到最大组合
    :param hole_cards:
    :param community_cards:
    :return:
    """
    all_cards = hole_cards + community_cards
    best_hand = all_cards[:5]
    max_hand_value = evaluate_hand(best_hand)

    # 获取所有从7张牌中选择5张的组合
    for combination in itertools.combinations(all_cards, 5):
        current_hand = list(combination)
        score1, hand_values1 = evaluate_hand(current_hand)
        score2, hand_values2 = max_hand_value

        win = compare_hands(score1, hand_values1, score2, hand_values2)

        if win == 1:
            best_hand = current_hand
            max_hand_value = [score1, hand_values1]

    return best_hand, max_hand_value


def get_winner(hole_cards1, hole_cards2, community_cards):
    best_hand1, max_hand_value1 = best_hand(hole_cards1, community_cards)
    best_hand2, max_hand_value2 = best_hand(hole_cards2, community_cards)
    return compare_hands(max_hand_value1[0], max_hand_value1[1], max_hand_value2[0],
                         max_hand_value2[1]), max_hand_value1, max_hand_value2


class simulation:
    def __init__(self, hole_cards, community_cards, iterations=1000):
        self.hole_cards = hole_cards
        self.community_cards = community_cards
        self.iterations = iterations
        self.deck_left = generate_deck()
        self.deck_left = [card for card in self.deck_left if card not in hole_cards and card not in community_cards]
        self.win_count = 0
        self.cards_value = [None] + [0 for _ in range(10)]

    def run(self):
        for iter in tqdm(range(self.iterations), desc='Iterations', colour='#4a65c4'):
            # 补全手牌
            self.deck_left = shuffle_deck(self.deck_left)  # 洗牌
            player2_hand = self.deck_left[:2]
            community_cards = self.community_cards.copy()
            community_cards += self.deck_left[2:7 - len(self.community_cards)]
            winner, value1, value2 = get_winner(self.hole_cards, player2_hand, community_cards)
            self.cards_value[value1[0]] += 1
            if winner == 1:
                self.win_count += 1


if __name__ == '__main__':

    # # 牌面
    # s = ['♠', '♣', '♦', '♥']
    # c = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
    #
    iterations = 50000
    # # data = {}
    # # for index, i in enumerate(c):
    # #     for j in c[index:]:
    # #         # 异色
    # #         hole_cards = [f'{i}{s[0]}', f'{j}{s[1]}']
    # #         community_cards = []
    # #         sim = simulation(hole_cards, community_cards, iterations)
    # #         sim.run()
    # #         data[(i, j, 2)] = [sim.win_count / sim.iterations, sim.cards_value]
    # #         if i != j:
    # #             data[(j, i, 2)] = [sim.win_count / sim.iterations, sim.cards_value]
    # #         # 同色
    # #         if i == j:
    # #             continue
    # #         hole_cards = [f'{i}{s[0]}', f'{j}{s[0]}']
    # #         community_cards = []
    # #         sim = simulation(hole_cards, community_cards, iterations)
    # #         sim.run()
    # #         data[(i, j, 1)] = [sim.win_count / sim.iterations, sim.cards_value]
    # #         data[(j, i, 1)] = [sim.win_count / sim.iterations, sim.cards_value]
    # #
    # # data = {str(key): value for key, value in data.items()}
    # # print(data)
    # #
    # # import json
    # #
    # # # 将字典保存为 JSON 格式的文件
    # # with open("my_data.json", "w") as f:
    # #     json.dump(data, f)
    #
    hole_cards = ['10♦', '10♥']
    community_cards = []
    sim = simulation(hole_cards, community_cards, iterations)
    sim.run()
    print(sim.win_count/iterations)

    hole_cards = ['9♦', '9♥']
    community_cards = []
    sim = simulation(hole_cards, community_cards, iterations)
    sim.run()
    print(sim.win_count / iterations)


