# 88.27%
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd


def decode(hands):
    rank_values = {
        '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10,
        'J': 11, 'Q': 12, 'K': 13, 'A': 14
    }
    suit_values = {
        '♠': 1, '♣': 2, '♦': 3, '♥': 4
    }
    array = np.zeros((len(hands), 4, 13))
    # 编码，按扑克值从大到小排序
    for i, hand in enumerate(hands):
        coded_cards = [(rank_values[card[:-1]] - 2, suit_values[card[-1]] - 1) for card in hand]
        for num in coded_cards:
            array[i, num[1], num[0]] = 1

    return array


if __name__ == '__main__':
    # 加载 .keras 格式的模型
    model = load_model("transformer_model.keras")

    # # 4 0 4 7 7 6 6 2 5 5
    # hands = [['2♠', '3♣', '6♦', '7♥', '8♣', '9♣', '10♣'],
    #          ['2♠', '3♣', '5♦', '7♥', '8♣', 'J♣', '10♣'],
    #          ['2♠', '3♣', '5♦', 'A♥', '8♣', 'J♣', '4♣'],
    #          ['A♠', '8♠', '8♦', '9♥', '9♠', '9♦', '9♣'],
    #          ['8♣', '8♠', '8♦', '9♥', '9♠', '9♦', '9♣'],
    #          ['A♠', '8♠', '8♦', '8♥', '9♠', '9♦', '9♣'],
    #          ['A♠', '8♠', 'A♦', '8♥', '9♠', '9♦', '9♣'],
    #          ['A♠', '8♠', 'A♦', '8♥', '10♠', '9♦', '9♣'],
    #          ['A♠', 'K♠', 'J♠', '8♦', '10♠', '9♠', '2♠'],
    #          ['A♠', 'K♠', 'J♠', '8♠', '10♠', '9♠', '2♠']]
    # X = decode(hands)
    # X = np.reshape(X, (-1, 4, 13))
    # predict = np.argmax(model.predict(X), axis=1)
    #
    # print(predict)

    data = pd.read_csv('tough_data.csv', header=0)


    def encode_card_data(row):
        # 直接返回每一行中牌的数据（4行7列）
        return np.array([[row[f"Card{i}_Suit"], row[f"Card{i}_Rank"]] for i in range(1, 8)])


    data_arr = np.array([encode_card_data(row) for _, row in data.iterrows()])

    X_data = np.zeros((len(data_arr), 4, 13))  # len(data_arr) 是样本数量

    for idx, datai in enumerate(data_arr):
        suits = datai[:, 0] - 1
        ranks = datai[:, 1] - 2
        X_data[idx, suits, ranks] = 1

    # 将 Hand_Result 转换为 one-hot 编码
    y_data = pd.get_dummies(data['Hand_Result']).values

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.4)

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_acc:.4f}")
