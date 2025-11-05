import os
import json
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    train_csv = os.path.join(data_dir, 'UNSW_NB15_training-set.csv')
    test_csv = os.path.join(data_dir, 'UNSW_NB15_testing-set.csv')

    if not (os.path.exists(train_csv) and os.path.exists(test_csv)):
        raise FileNotFoundError('Training/testing CSV not found under data/.')

    train = pd.read_csv(train_csv, index_col='id')
    test = pd.read_csv(test_csv, index_col='id')

    unsw = pd.concat([train, test], axis=0)
    unsw = pd.get_dummies(data=unsw, columns=['proto', 'service', 'state'])

    # Save the full feature columns after dropping labels
    feature_df = unsw.drop(['label', 'attack_cat'], axis=1)
    feature_columns = list(feature_df.columns)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(feature_df.values)

    # Persist
    with open(os.path.join(data_dir, 'feature_columns.json'), 'w', encoding='utf-8') as f:
        json.dump({'feature_columns': feature_columns}, f, ensure_ascii=False, indent=2)

    joblib.dump(scaler, os.path.join(data_dir, 'scaler.pkl'))

    print('Saved:')
    print(' -', os.path.join(data_dir, 'feature_columns.json'))
    print(' -', os.path.join(data_dir, 'scaler.pkl'))


if __name__ == '__main__':
    main()

