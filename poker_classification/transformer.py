# 86.53%
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.layers import Input, MultiHeadAttention, Dropout, LayerNormalization, Conv1D, GlobalAveragePooling1D, Dense
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Flatten

# ---------------------------------------------数据清洗---------------------------------------------
data = pd.read_csv('poker_hand_data.csv', header=0)


def encode_card_data(row):
    # 直接返回每一行中牌的数据（4行7列）
    return np.array([[row[f"Card{i}_Suit"], row[f"Card{i}_Rank"]] for i in range(1, 8)])

# 处理每一行的牌，生成 3D 数组
data_arr = np.array([encode_card_data(row) for _, row in data.iterrows()])

# 初始化全零数组并填充
X_data = np.zeros((len(data_arr), 4, 13))  # len(data_arr) 是样本数量

for idx, datai in enumerate(data_arr):
    suits = datai[:, 0] - 1
    ranks = datai[:, 1] - 2
    X_data[idx, suits, ranks] = 1

# 将 Hand_Result 转换为 one-hot 编码
y_data = pd.get_dummies(data['Hand_Result']).values

# ---------------------------------------------模型搭建---------------------------------------------

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = Dropout(dropout)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    x = Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = Dropout(dropout)(x)
    x = Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    return x + res

def transformer_model(input_shape=(4, 13), num_classes=10, lr=0.001, head_size=128, num_heads=4, ff_dim=4, num_transformer_blocks=4, mlp_units=[64], dropout=0.2, mlp_dropout=0.2):
    inputs = Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = Flatten()(x)
    for dim in mlp_units:
        x = Dense(dim, activation="relu")(x)
        x = Dropout(mlp_dropout)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=lr),
        metrics=['accuracy']
    )
    return model

model = transformer_model(input_shape=(4, 13), num_classes=y_data.shape[1])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

# 训练模型
early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
model.fit(X_train, y_train, epochs=500, batch_size=64, validation_data=(X_test, y_test), callbacks=[early_stopping])


# 评估模型
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test loss: {test_loss}, Test accuracy: {test_acc}")

model.save("transformer_model.keras")
