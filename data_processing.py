import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def load_data():
    # Default CSV paths expected under data/
    train_csv = 'data/UNSW_NB15_training-set.csv'
    test_csv = 'data/UNSW_NB15_testing-set.csv'

    # Read CSVs and set 'id' as index
    train = pd.read_csv(train_csv, index_col='id')  # 指定“id”列作为行索引
    test = pd.read_csv(test_csv, index_col='id')    # 指定“id”列作为行索引

    # Binary labels
    training_label = train['label'].values  # 训练集标签
    testing_label = test['label'].values    # 测试集标签

    # Concatenate for consistent one-hot across splits
    unsw = pd.concat([train, test])  # 先合并再编码，保持列一致
    unsw = pd.get_dummies(data=unsw, columns=['proto', 'service', 'state'])  # 类别特征 one-hot 编码

    # Drop non-feature columns and scale numeric features
    unsw.drop(['label', 'attack_cat'], axis=1, inplace=True)  # 删除标签列
    unsw_value = unsw.values

    scaler = MinMaxScaler(feature_range=(0, 1))  # 归一化到 [0,1]
    unsw_value = scaler.fit_transform(unsw_value)

    # Split back to train/test
    train_set = unsw_value[:len(train), :]
    test_set = unsw_value[len(train):, :]

    return train_set, training_label, test_set, testing_label
