# 91.87%
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd

# 读取数据
data = pd.read_csv('poker_winner_data.csv', header=0)

data_arr = data.iloc[:, :-1].to_numpy()

X_data = np.zeros((len(data_arr), 4, 13, 2))  # 两通道
for index, row in enumerate(data_arr):
    X_data[index, row[0] - 1, row[1] - 2, 0] = 1
    X_data[index, row[2] - 1, row[3] - 2, 0] = 1
    X_data[index, row[4] - 1, row[5] - 2, 1] = 1
    X_data[index, row[6] - 1, row[7] - 2, 1] = 1
    for j in range(5):
        X_data[index, row[2 * j + 8] - 1, row[2 * j + 9] - 2, 0] = 1
        X_data[index, row[2 * j + 8] - 1, row[2 * j + 9] - 2, 1] = 1

# One-hot 编码目标
y_data = pd.get_dummies(data['winner']).values

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

# 参数设置
learning_rate = 0.001
epochs = 200
batch_size = 64
num_classes = y_data.shape[1]  # 分类数

# 使用 tf.keras 构建模型
model = tf.keras.Sequential([
    # 卷积层 1
    tf.keras.layers.Conv2D(32, (2, 2), activation='relu', padding='same', input_shape=(4, 13, 2)),
    tf.keras.layers.MaxPooling2D((2, 2), padding='same'),

    # 卷积层 2
    tf.keras.layers.Conv2D(64, (2, 2), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2), padding='same'),

    # 展平层
    tf.keras.layers.Flatten(),

    # 全连接层 1
    tf.keras.layers.Dense(128, activation='relu'),

    # Dropout 层 (全连接层 1 后)
    tf.keras.layers.Dropout(0.5),

    # 全连接层 2
    tf.keras.layers.Dense(64, activation='relu'),

    # 输出层
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# 编译模型
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 早停
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor='val_loss',  # 根据验证集损失判断
    patience=40,  # 连续 40 轮没有提升后停止
    restore_best_weights=True
)

# 设置保存模型的回调函数，只保存验证集上性能最好的模型
from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(
    filepath='best_cnn_model.h5',  # 保存路径
    monitor='val_accuracy',  # 监控的指标，可根据需要改为 'val_loss'
    save_best_only=True,  # 仅保存最佳模型
    mode='max',  # 指定指标越大越好（对于 accuracy）
    verbose=1,  # 在每次保存时输出日志
)

# 训练模型
history = model.fit(
    X_train,
    y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.2,
    callbacks=[checkpoint_callback, early_stopping]  # 添加回调
)

# 测试模型
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

model.save('cnn_model.keras')
