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

features_avg = ['B_1', 'B_11', 'B_16', 'B_17', 'B_18', 'B_2', 'B_20',
                'B_28', 'B_3', 'B_4', 'B_5', 'B_7', 'B_9', 'D_112',
                'D_121', 'D_141', 'D_39', 'D_41', 'D_42', 'D_43',
                'D_44', 'D_45', 'D_46', 'D_47', 'D_48', 'D_49', 
                'D_50', 'D_51', 'D_53', 'D_54', 'D_56', 'D_58', 
                'D_59', 'D_60', 'D_91', 'P_2', 'P_3', 'R_1', 'R_2', 
                'R_27', 'R_3', 'R_7', 'S_11', 'S_26', 'S_3', 'S_5']
features_last = ['B_1', 'B_10', 'B_11', 'B_12', 'B_13', 'B_15', 'B_16',
                 'B_17', 'B_18', 'B_19', 'B_2', 'B_20', 'B_22', 'B_23',
                 'B_24', 'B_25', 'B_26', 'B_27', 'B_28', 'B_29', 'B_3',
                 'B_32', 'B_33', 'B_36', 'B_38', 'B_39', 'B_4', 'B_40',
                 'B_41', 'B_42', 'B_5', 'B_6', 'B_7', 'B_8', 'B_9',
                 'D_102', 'D_103', 'D_105', 'D_106', 'D_107', 'D_109',
                 'D_112', 'D_115', 'D_117', 'D_118', 'D_119', 'D_120',
                 'D_121', 'D_122', 'D_123', 'D_124', 'D_125', 'D_127', 
                 'D_129', 'D_132', 'D_133', 'D_135', 'D_136', 'D_137', 
                 'D_140', 'D_141', 'D_143', 'D_145', 'D_39', 'D_41',
                 'D_42', 'D_43', 'D_44', 'D_45', 'D_46', 'D_47', 'D_48',
                 'D_49', 'D_50', 'D_51', 'D_52', 'D_53', 'D_54', 'D_55',
                 'D_56', 'D_58', 'D_59', 'D_60', 'D_61', 'D_62', 'D_63',
                 'D_64', 'D_66', 'D_70', 'D_72', 'D_73', 'D_74', 'D_75',
                 'D_76', 'D_77', 'D_78', 'D_79', 'D_80', 'D_82', 'D_83',
                 'D_84', 'D_86', 'D_91', 'D_92', 'D_93', 'D_94', 'D_96',
                 'P_2', 'P_3', 'R_1', 'R_10', 'R_11', 'R_12', 'R_13',
                 'R_14', 'R_15', 'R_17', 'R_18', 'R_19', 'R_2', 'R_20', 
                 'R_21', 'R_22', 'R_24', 'R_25', 'R_26', 'R_27', 'R_3',
                 'R_4', 'R_5', 'R_7', 'R_8', 'R_9', 'S_11', 'S_12',
                 'S_13', 'S_15', 'S_17', 'S_20', 'S_22', 'S_23', 
                 'S_24', 'S_25', 'S_26', 'S_27', 'S_3', 'S_5', 'S_6',
                 'S_7', 'S_8', 'S_9']
cat_features = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']

def make_categorical_features(df):
    cat_df = pd.DataFrame([], index=df['customer_ID'].unique())
    for feature in cat_features:
        cl_sv = pd.pivot_table(df, index='customer_ID', columns=feature, values = 'P_2', aggfunc = 'count')
        cl_sv['summs'] = cl_sv.sum(axis=1)
        for i in cl_sv.columns[:-1]:
            cl_sv[i] = cl_sv[i] / cl_sv['summs']
        cl_sv.columns = [feature + '_'+ str(i) + '_count' for i in cl_sv.columns]
        cat_df = cat_df.merge(cl_sv, how='left', left_index=True, right_index=True)

    nunique_df = df.groupby('customer_ID')[cat_features].nunique()
    nunique_df.columns = [f'{x}_nunique' for x in nunique_df.columns]

    cat_df = cat_df.merge(nunique_df, how='left', left_index=True, right_index=True).sort_index().fillna(0)
    del cl_sv
    return cat_df


def get_data(path, train=False):
    data = pd.read_feather(path)
    cat_df = make_categorical_features(data)
    if 'target' in data.columns:
        main_data = data.drop('target', axis=1).groupby('customer_ID').agg(['mean', 'median', 'std', 'min', 'max']).sort_index()
    else:
        main_data = data.groupby('customer_ID').agg(['mean', 'median', 'std', 'min', 'max']).sort_index()
    main_data.columns = [f'{x[0]}_{x[1]}' for x in main_data.columns]
    cid = pd.Categorical(data.pop('customer_ID'), ordered=True)
    last = (cid != np.roll(cid, -1))
    if train: target = data.loc[last, 'target']
    df_avg = data[features_avg].groupby(cid).mean().rename(columns={f: f"{f}_avg" for f in features_avg})
    data = data.loc[last, features_last].rename(columns={f: f"{f}_last" for f in features_last}).set_index(np.asarray(cid[last]))
    data = pd.concat([data, df_avg, cat_df, main_data], axis=1)
    cat_features = data.select_dtypes(['category']).columns.tolist()
    data[cat_features] = data[cat_features].astype('str')

    if train:
        encoders = []
        for feature in cat_features:
            cat_encoder = LabelEncoder()
            data[feature] = cat_encoder.fit_transform(data[feature])
            encoders.append(cat_encoder)
        with open('models/cat_encoder.pkl', 'wb') as f:
            pickle.dump(encoders, f)
    else:
        with open('models/cat_encoder.pkl', 'rb') as f:
            encoders = pickle.load(f)
        for i in range(len(cat_features)):
            data[cat_features[i]] = encoders[i].transform(data[cat_features[i]])
    
    gc.collect()

    if train: return data, target, cat_features
    else: return data


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
