"""
99.37%
Best Hyperparameters:
{'learning_rate': 0.0019678693041818413,
'dropout_rate': 0.35,
'filters_1': 32,
'filters_2': 128,
'kernel_size': 3,
'units_1': 256,
'units_2': 64,
'tuner/epochs': 50,
'tuner/initial_epoch': 17,
'tuner/bracket': 2,
'tuner/round': 2,
'tuner/trial_id': '0062'}
"""
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
import keras_tuner as kt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def hands_to_tensor(hands):
    """
    将手牌组变为(-1, 4, 13, 1)张量
    :param hands: eg. [['2♠', '3♣', '6♦', '7♥', '8♣', '9♣', '10♣'],
                       ['2♠', '3♣', '5♦', '7♥', '8♣', 'J♣', '10♣'],
                       ['2♠', '3♣', '5♦', 'A♥', '8♣', 'J♣', '4♣']]

    :return: (-1, 4, 13, 1)张量
    """
    rank_values = {
        '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10,
        'J': 11, 'Q': 12, 'K': 13, 'A': 14
    }
    suit_values = {
        '♠': 1, '♣': 2, '♦': 3, '♥': 4
    }
    X = np.zeros((len(hands), 4, 13, 1))

    for i, hand in enumerate(hands):
        coded_cards = [(rank_values[card[:-1]] - 2, suit_values[card[-1]] - 1) for card in hand]
        for num in coded_cards:
            X[i, num[1], num[0], 0] = 1

    return X


def csv_to_tensor(data):
    """
    将csv格式numpy数组变为(-1, 4, 13, 1)张量
    :param data: numpy格式(-1,15)数组,包含最后一列
    :return: X (张量), y (one-hot编码)
    """
    # 拆分输入特征和目标
    X_data = data[:, :-1]  # 前14列为特征
    y_data = data[:, -1]  # 最后一列为标签

    # 提取花色和点数
    suits = X_data[:, ::2] - 1  # 花色 (奇数列)
    ranks = X_data[:, 1::2] - 2  # 点数 (偶数列)

    # 初始化 X 并使用广播赋值
    X = np.zeros((len(data), 4, 13, 1))
    for idx in range(7):  # 每组样本最多有7对花色和点数
        X[np.arange(len(data)), suits[:, idx], ranks[:, idx], 0] = 1

    # One-hot 编码目标
    y = pd.get_dummies(y_data).to_numpy()

    return X, y


def tensor_to_hands(X):
    """
    将(-1, 4, 13, 1)张量变为手牌组形式
    :param X: (-1, 4, 13, 1)张量
    :return: hands, eg. [['2♠', '3♣', '6♦', '7♥', '8♣', '9♣', '10♣'],
                        ['2♠', '3♣', '5♦', '7♥', '8♣', 'J♣', '10♣'],
                        ['2♠', '3♣', '5♦', 'A♥', '8♣', 'J♣', '4♣']]
    """

    rank_values = {
        '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10,
        'J': 11, 'Q': 12, 'K': 13, 'A': 14
    }
    suit_values = {
        '♠': 1, '♣': 2, '♦': 3, '♥': 4
    }

    # 反向映射
    reverse_rank_values = {v: k for k, v in rank_values.items()}
    reverse_suit_values = {v: k for k, v in suit_values.items()}

    if X.ndim == 4:
        hands = []

        # 遍历每个样本
        for sample in X:
            hand = []
            # 查找非零值的位置
            indices = np.argwhere(sample[:, :, 0] == 1)
            for suit_idx, rank_idx in indices:
                # 根据索引还原花色和点数
                suit = reverse_suit_values[suit_idx + 1]
                rank = reverse_rank_values[rank_idx + 2]
                # 拼接成手牌字符串
                hand.append(f"{rank}{suit}")
            hands.append(hand)

        return hands

    elif X.ndim == 3:
        hand = []
        # 查找非零值的位置
        indices = np.argwhere(X[:, :, 0] == 1)
        for suit_idx, rank_idx in indices:
            # 根据索引还原花色和点数
            suit = reverse_suit_values[suit_idx + 1]
            rank = reverse_rank_values[rank_idx + 2]
            # 拼接成手牌字符串
            hand.append(f"{rank}{suit}")
        return hand


def create_model(hp):
    """
    创建 Keras 模型并允许传递超参数
    :param hp: 超参数对象
    :return: 编译好的模型
    """
    # 调优的超参数
    learning_rate = hp.Float('learning_rate', min_value=1e-6, max_value=1e-2, default=1e-3, sampling='log')
    dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.05, default=0.5)
    filters_1 = hp.Int('filters_1', min_value=32, max_value=128, step=32, default=32)
    filters_2 = hp.Int('filters_2', min_value=64, max_value=256, step=64, default=64)
    kernel_size = hp.Int('kernel_size', min_value=2, max_value=3, default=2)
    units_1 = hp.Int('units_1', min_value=64, max_value=256, step=64, default=128)
    units_2 = hp.Int('units_2', min_value=32, max_value=128, step=32, default=64)

    # 构建模型
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters_1, (kernel_size, kernel_size), activation='relu', padding='same',
                               input_shape=(4, 13, 1)),
        tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
        tf.keras.layers.Conv2D(filters_2, (kernel_size, kernel_size), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units_1, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(units_2, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    # 编译模型
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


if __name__ == '__main__':
    # --------------------------------------------数据清洗--------------------------------------------

    # 读取数据
    data = pd.read_csv('poker_hand_data.csv', header=0)
    X_data, y_data = csv_to_tensor(data.to_numpy())

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

    # 设置类别数
    num_classes = y_data.shape[1]

    # --------------------------------------------贝叶斯优化--------------------------------------------

    # 创建 Keras Tuner 调优对象
    tuner = kt.Hyperband(
        create_model,
        objective='val_accuracy',
        max_epochs=50,
        factor=3,
        directory='hyperband',
        project_name='poker_hand'
    )

    # 设置回调
    early_stopping = EarlyStopping(
        monitor='val_loss',  # 根据验证集损失判断
        patience=100,  # 连续 100 轮没有提升后停止
        restore_best_weights=True
    )

    checkpoint_callback = ModelCheckpoint(
        filepath='best_cnn_model2.h5',
        monitor='val_accuracy',  # 监控的指标 'val_accuracy' or 'val_loss'
        save_best_only=True,
        mode='max',  # 最大化指标
        verbose=1,  # 保存时输出日志
    )

    # 训练模型并进行超参数调优
    tuner.search(X_train, y_train, epochs=50, validation_data=(X_test, y_test),
                 callbacks=[checkpoint_callback, early_stopping])

    # 获取最佳超参数
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("Best Hyperparameters:", best_hps.values)

    # 使用最佳超参数重新训练模型
    model = tuner.hypermodel.build(best_hps)
    model.fit(X_train, y_train, epochs=400, batch_size=64, validation_split=0.2,
              callbacks=[checkpoint_callback, early_stopping])


