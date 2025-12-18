from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd


def encode(hole, oppo, comu):
    rank_values = {
        '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10,
        'J': 11, 'Q': 12, 'K': 13, 'A': 14
    }
    suit_values = {
        '♠': 1, '♣': 2, '♦': 3, '♥': 4
    }
    array = np.zeros((1, 4, 13, 2))
    for card in hole:
        array[0, suit_values[card[-1]] - 1, rank_values[card[:-1]] - 2, 0] = 1
    for card in oppo:
        array[0, suit_values[card[-1]] - 1, rank_values[card[:-1]] - 2, 1] = 1
    for card in comu:
        array[0, suit_values[card[-1]] - 1, rank_values[card[:-1]] - 2, 0] = 1
        array[0, suit_values[card[-1]] - 1, rank_values[card[:-1]] - 2, 1] = 1

    return array


if __name__ == '__main__':
    # 加载 .keras 格式的模型
    cnn_model = load_model('cnn_model.keras')

    hole = ['2♠', '3♣']
    oppo = ['8♦', '8♥']
    comu = ['5♦', '6♥', '8♣', '7♠', '9♣']

    X = encode(hole, oppo, comu)

    # 使用加载的模型进行预测或继续训练
    predict = np.argmax(cnn_model.predict(X), axis=1)
    print(predict)

