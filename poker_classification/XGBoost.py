# 46.94%
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# 读取 CSV 文件
data = pd.read_csv("poker_hand_data.csv")

# 特征数据 (X) 和标签数据 (y)
X = data.drop(columns=["Hand_Result"])
y = data["Hand_Result"]

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 创建 XGBoost 分类器
xgb_clf = XGBClassifier(objective='multi:softmax', num_class=len(y.unique()), random_state=42)

# 定义参数网格
param_grid = {
    'max_depth': [6, 12],         # 树的深度
    'min_child_weight': [1, 5],  # 最小叶子节点权重
    'gamma': [0.1, 0.3],         # 分裂所需最小损失下降值
    'subsample': [0.8, 1],          # 数据采样比例
    'colsample_bytree': [0.8, 1],   # 特征采样比例
    'eta': [0.1, 0.3],              # 学习率
}

# 使用 GridSearchCV 搜索最佳参数
grid_search = GridSearchCV(estimator=xgb_clf, param_grid=param_grid, scoring='accuracy', cv=3, verbose=2)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)

best_xgb = grid_search.best_estimator_
best_xgb.fit(X_train, y_train)

y_pred = best_xgb.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"XGBoost Model Accuracy with Best Parameters: {accuracy * 100:.2f}%")

# # 可视化特征重要性
# xgb.plot_importance(best_xgb)
# plt.show()