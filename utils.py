from nis import cat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import ListedColormap
import seaborn as sns
from cycler import cycler
from IPython.display import display
import datetime
import warnings
from colorama import Fore, Style
import gc
import pickle

from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibrationDisplay
import lightgbm
from catboost import Pool, CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
import catboost


def make_num_features(df, num_columns):
    new_data = df[num_columns+['customer_ID']].groupby('customer_ID').agg(['mean', 'median', 'std', 'min', 'max'])
    new_data.columns = [f'{x[0]}_{x[1]}' for x in new_data.columns]
    return new_data


def make_categorical_features(df):
    new_data = df.groupby('customer_ID').agg(['nunique'])
    new_data.columns = [f'{x[0]}_{x[1]}' for x in new_data.columns]
    return new_data


def make_last_features(df, num_cols, cat_cols):
    cid = df['customer_ID']
    last = (cid != np.roll(cid, -1))
    data_last = df.loc[last, num_cols+cat_cols].rename(columns={f: f"{f}_last" for f in num_cols+cat_cols}).set_index(np.asarray(cid[last]))
    return data_last, [f'{x}_last' for x in cat_cols]


def get_data(path, train=False):
    data = pd.read_feather(path)
    data.rename({'S_2': 'date'}, axis=1, inplace=True)


    """
    Не использовалось
    for col in data.columns:
        if data[col].dtype=='float16':
            data[col] = data[col].astype('float32').round(decimals=2).astype('float16')
    """

    info_columns = ['date', 'target', 'customer_ID']
    bin_columns = ['D_87', 'B_31']
    cat_columns = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']
    num_columns = sorted([x for x in data.columns if x not in info_columns+bin_columns+cat_columns])

    if train:
        encoders = []
        for feature in cat_columns:
            cat_encoder = LabelEncoder()
            data[feature] = cat_encoder.fit_transform(data[feature])
            encoders.append(cat_encoder)
        with open('models/cat_encoder.pkl', 'wb') as f:
            pickle.dump(encoders, f)
    else:
        with open('models/cat_encoder.pkl', 'rb') as f:
            encoders = pickle.load(f)
        for i in range(len(cat_columns)):
            data[cat_columns[i]] = encoders[i].transform(data[cat_columns[i]])

    cat_features = make_categorical_features(data[cat_columns + ['customer_ID']])
    num_features = make_num_features(data, num_columns)
    last_features, cat_cols = make_last_features(data, num_columns, cat_columns)
    all_features = cat_features.merge(num_features, how='left', left_index=True, right_index=True).merge(
                                        last_features, how='left', left_index=True, right_index=True).sort_index()
    del cat_features, num_features, last_features
    gc.collect()

    if train:
        return all_features, data[['customer_ID', 'target']].drop_duplicates().sort_values('customer_ID')['target'], cat_cols
    else:
        return all_features


def amex_metric(y_true, y_pred, return_components=False):
    def top_four_percent_captured(df):
        df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
        four_pct_cutoff = int(0.04 * df['weight'].sum())
        df['weight_cumsum'] = df['weight'].cumsum()
        df_cutoff = df.loc[df['weight_cumsum'] <= four_pct_cutoff]
        return (df_cutoff['target'] == 1).sum() / (df['target'] == 1).sum()
        
    def weighted_gini(df):
        df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
        df['random'] = (df['weight'] / df['weight'].sum()).cumsum()
        total_pos = (df['target'] * df['weight']).sum()
        df['cum_pos_found'] = (df['target'] * df['weight']).cumsum()
        df['lorentz'] = df['cum_pos_found'] / total_pos
        df['gini'] = (df['lorentz'] - df['random']) * df['weight']
        return df['gini'].sum()

    def normalized_weighted_gini(df):
        """Corresponds to 2 * AUC - 1"""
        df2 = pd.DataFrame({'target': df.target, 'prediction': df.target})
        df2.sort_values('prediction', ascending=False, inplace=True)
        return weighted_gini(df) / weighted_gini(df2)

    df = pd.DataFrame({'target': y_true.ravel(), 'prediction': y_pred.ravel()})
    df.sort_values('prediction', ascending=False, inplace=True)
    g = normalized_weighted_gini(df)
    d = top_four_percent_captured(df)

    if return_components: return g, d, 0.5 * (g + d)
    return 0.5 * (g + d)

def lgb_amex_metric(y_true, y_pred):
    """The competition metric with lightgbm's calling convention"""
    return ('amex',
            amex_metric(y_true, y_pred),
            True)
