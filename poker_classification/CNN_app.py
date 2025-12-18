# 98.78%
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from CNN import csv_to_tensor, hands_to_tensor, tensor_to_hands
import random

if __name__ == '__main__':
    # ------------------------------------模型加载-------------------------------------

    cnn_model = load_model("best_cnn_model.h5")

    # ------------------------------------自定义测试------------------------------------

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
    # X = hands_to_tensor(hands)
    #
    # # 预测
    # predict = np.argmax(cnn_model.predict(X), axis=1)
    # print(predict)

    # ----------------------------------tough_data测试----------------------------------

    # 99.5%
    data = pd.read_csv('tough_data.csv', header=0)

    # 数据预处理
    X, y = csv_to_tensor(data.to_numpy())

    # 测试模型
    predict = np.argmax(cnn_model.predict(X), axis=-1)
    discrepancy = predict - np.argmax(y, axis=-1)
    for i, n in enumerate(discrepancy):
        if n != 0:
            print(predict[i], np.argmax(y, axis=-1)[i], tensor_to_hands(X[i]))
    print(np.sum(discrepancy == 0) / len(discrepancy))

    # test_loss, test_acc = cnn_model.evaluate(X, y)
    # print(f"Test Accuracy: {test_acc:.4f}")

    # -------------------------------------随机测试-------------------------------------

    # 98.78%
    from simulation import generate_deck, evaluate

    deck = generate_deck()
    iterations = 100000
    hands = []
    y_data = []
    for i in range(iterations):
        random.shuffle(deck)
        hole = deck[:7]
        hands.append(hole)
        score, _ = evaluate(hole, [])
        y_data.append(score)

    # predict
    X = hands_to_tensor(hands)
    y = pd.get_dummies(y_data).to_numpy()
    test_loss, test_acc = cnn_model.evaluate(X, y)
    print(f"Test Accuracy: {test_acc:.4f}")


