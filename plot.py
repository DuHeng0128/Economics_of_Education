import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import pyreadstat
from scipy.spatial.distance import squareform, pdist
import statsmodels.api as sm
from linearmodels import PanelOLS
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from sklearn.preprocessing import MinMaxScaler
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False 

def peer_weight(sub_df):

    scores = sub_df[['G']].values
    weights = 1 / (1 + squareform(pdist(scores)))
    np.fill_diagonal(weights, 0)
    return weights / weights.sum(axis=1)[:, None]

def standardize_and_mean(df, columns):

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[columns])
    return scaled_data.mean(axis=1)

def ces_function(X, alpha, beta, gamma, p):
    G, M, peer_E = X
    eps = 1e-10
    
    beta = 1.0 - alpha if alpha + beta > 1.0 else beta
    
    term = alpha * (G + eps)**p + beta * (M + eps)**p
    term = np.clip(term, 1e-10, None)
    
    exp_term = gamma * peer_E
    exp_term = np.clip(exp_term, -100, 100)
    return term**(1/(p + eps)) * np.exp(exp_term)


if __name__ == "__main__":
    df, _ = pyreadstat.read_sav("Education_data\eductaion_data.sav")
    scaler = MinMaxScaler(feature_range=(0, 1))

    df['stdchn'] = df['stdchn'].apply(lambda x: x[0] if isinstance(x, np.ndarray) else x)
    df['stdmat'] = df['stdmat'].apply(lambda x: x[0] if isinstance(x, np.ndarray) else x)
    df['stdeng'] = df['stdeng'].apply(lambda x: x[0] if isinstance(x, np.ndarray) else x)
    df['G_raw'] = df[['stdchn', 'stdmat', 'stdeng']].mean(axis=1, skipna=False)
    df['G'] = scaler.fit_transform(df[['G_raw']])

    mental_health_items = ['a1801', 'a1802', 'a1803', 'a1804', 'a1805']
    df['M_raw'] = df[mental_health_items].sum(axis=1)
    df['M'] = scaler.fit_transform(df[['M_raw']])

    df['S_family'] = standardize_and_mean(df, 
        ['b2501', 'b2502', 'b24a1', 'b24b1', 'b24a2', 'b24b2', 'b24a4', 'b24b4'])
    df['S_family_pressure'] = standardize_and_mean(df,
        ['b2301', 'b2302'])
    df['S_peer'] = standardize_and_mean(df,
        ['c1710', 'c19'])
    df['S'] = df[['S_family', 'S_family_pressure','S_peer']].mean(axis=1)

    #df['cog3pl'] = scaler.fit_transform(df[['cog3pl']])

    df['peer_E'] = df[['c2101', 'c2102']].mean(axis=1)

    columns_to_check = ['G', 'S', 'M', 'peer_E', 'cog3pl']
    df = df.dropna(subset=columns_to_check, how='any')

    """class_groups = df.groupby('clsids')
    class_weights = {}
    for class_id, class_df in class_groups:
        class_weights[class_id] = peer_weight(class_df)"""

    class_groups = df.groupby('clsids')
    df['peer_E_weighted'] = 0.0
    for class_id, class_df in class_groups:
        weights = peer_weight(class_df)
        E_values = class_df['peer_E'].values
        df.loc[class_df.index, 'peer_E_weighted'] = np.dot(weights, E_values)


    sns.kdeplot(x=df.G, y=df.M, cmap="viridis", fill=True)
    plt.title("G-M替代曲面")
    plt.xlabel("成绩产出G")
    plt.ylabel("心理健康M")
    plt.savefig('G_M_surface.png')

    sns.lmplot(x='S', y='G', hue='peer_E_quantile', 
            data=df.assign(peer_E_quantile=pd.qcut(df.peer_E_weighted, 5, duplicates='drop')))
    plt.title("社会资本S对成绩产出的调节效应")
    plt.savefig('S_G_interaction.png')