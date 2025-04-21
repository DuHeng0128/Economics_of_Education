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

def peer_weight(sub_df):
    """距离函数（示例：基于标准化成绩的相似性）"""
    scores = sub_df[['G']].values
    weights = 1 / (1 + squareform(pdist(scores)))
    np.fill_diagonal(weights, 0)
    return weights / weights.sum(axis=1)[:, None]

def standardize_and_mean(df, columns):
    """对指定列进行标准化后计算行均值"""
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[columns])
    return scaled_data.mean(axis=1)

def ces_function(X, alpha, beta, gamma, p):
    """带数值保护的CES函数"""
    G, M, peer_E = X
    eps = 1e-10
    
    # 约束 alpha + beta ≈ 1
    beta = 1.0 - alpha if alpha + beta > 1.0 else beta
    
    # 计算核心项（添加极小值保护）
    term = alpha * (G + eps)**p + beta * (M + eps)**p
    term = np.clip(term, 1e-10, None)  # 防止负数开根
    
    # 处理指数项溢出
    exp_term = gamma * peer_E
    exp_term = np.clip(exp_term, -100, 100)  # 防止exp爆炸
    return term**(1/(p + eps)) * np.exp(exp_term)

def exam_nan_inf(X_data, y_data):
    X = np.array(X_data)
    y = np.array(y_data).ravel()

    print("X_data 非有限值总数:", np.isfinite(X).size - np.isfinite(X).sum())
    print("y_data 非有限值总数:", np.isfinite(y).size - np.isfinite(y).sum())

    # 如果你想知道具体是哪几行：
    bad_X = ~np.isfinite(X).all(axis=1)
    bad_y = ~np.isfinite(y)
    print("X 中含非有限值的行索引:", np.where(bad_X)[0])
    print("y 中含非有限值的行索引:", np.where(bad_y)[0])


def fit_ces_model(X_data, y_data):
    """拟合CES模型"""
    initial_guess = [0.6, 0.4, 0.01, 1.0]  # [alpha, beta, gamma, p]
    bounds = (
        [0.1, 0.1, -np.inf, 0.5],   # 下限
        [0.9, 0.9, np.inf, 2.0]      # 上限
    )
    #exam_nan_inf(X_data, y_data)

    params, covariance = curve_fit(
        ces_function,
        X_data,
        y_data,
        p0=initial_guess,
        bounds=bounds,  # 参数约束
        maxfev=10000,
        method='trf'  # 使用信赖区域反射法
    )

    # 输出结果
    print("估计参数:")
    print(f"alpha = {params[0]:.4f}")
    print(f"beta = {params[1]:.4f}")
    print(f"gamma = {params[2]:.4f}")
    print(f"p = {params[3]:.4f}")

    # 计算R²
    y_pred = ces_function(X_data, *params)
    ss_res = np.sum((y_data - y_pred)**2)
    ss_tot = np.sum((y_data - np.mean(y_data))**2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f"\nR² = {r_squared:.4f}")

    # 参数标准误计算（使用协方差矩阵对角元素）
    perr = np.sqrt(np.diag(covariance))
    print("\n参数标准误:")
    print(f"alpha: {perr[0]:.4f}")
    print(f"beta: {perr[1]:.4f}")
    print(f"gamma: {perr[2]:.4f}")
    print(f"p: {perr[3]:.4f}")

    return params, covariance, y_pred


if __name__ == "__main__":
    # 读取数据
    df, _ = pyreadstat.read_sav("Education_data\eductaion_data.sav")
    scaler = MinMaxScaler(feature_range=(0, 1))

    # 构建核心变量
    # 成绩产出G（标准化成绩均值）
    df['stdchn'] = df['stdchn'].apply(lambda x: x[0] if isinstance(x, np.ndarray) else x)
    df['stdmat'] = df['stdmat'].apply(lambda x: x[0] if isinstance(x, np.ndarray) else x)
    df['stdeng'] = df['stdeng'].apply(lambda x: x[0] if isinstance(x, np.ndarray) else x)
    df['G_raw'] = df[['stdchn', 'stdmat', 'stdeng']].mean(axis=1, skipna=False)
    df['G'] = scaler.fit_transform(df[['G_raw']])

    # 心理健康M（负面情绪综合指标）
    mental_health_items = ['a1801', 'a1802', 'a1803', 'a1804', 'a1805']
    df['M_raw'] = df[mental_health_items].sum(axis=1)
    df['M'] = scaler.fit_transform(df[['M_raw']])

    # 社会资本S（家庭社会资本+同伴社会资本）
    df['S_family'] = standardize_and_mean(df, 
        ['b2501', 'b2502', 'b24a1', 'b24b1', 'b24a2', 'b24b2', 'b24a4', 'b24b4'])
    df['S_family_pressure'] = standardize_and_mean(df,
        ['b2301', 'b2302'])
    df['S_peer'] = standardize_and_mean(df,
        ['c1710', 'c19'])
    df['S'] = df[['S_family', 'S_family_pressure','S_peer']].mean(axis=1)

    #df['cog3pl'] = scaler.fit_transform(df[['cog3pl']])

    # 同伴效应E
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

    # CES生产函数回归
    X = df[['G', 'M', 'peer_E_weighted']]
    X = sm.add_constant(X)
    y = df['cog3pl']
    model = sm.OLS(y, X).fit()
    print(model.summary())
    
    # 估计替代弹性参数p（需要非线性回归）
    X_data = [
        df['G'].values.astype(np.float64),
        df['M'].values.astype(np.float64),
        df['peer_E_weighted'].values.astype(np.float64)
    ]
    y_data = df['cog3pl'].values.astype(np.float64)

    """print("数据统计摘要:")
    print(df[['G', 'M', 'peer_E_weighted', 'cog3pl']].describe())
    print("\n无穷值统计:")
    print(np.isinf(df[['G', 'M', 'peer_E_weighted', 'cog3pl']]).sum())
    print("\nNaN统计:")
    print(np.isnan(df[['G', 'M', 'peer_E_weighted', 'cog3pl']]).sum())"""

    params, covariance, y_pred = fit_ces_model(X_data, y_data)
    df['correction'] = pd.DataFrame(y_pred, columns=["correction"], index=df.index)


    
    df['family_background'] = standardize_and_mean(df, ['steco_5c'])
    df['policy'] = standardize_and_mean(df, ['c18'])
    df['courses'] = standardize_and_mean(df, ['c1301', 'c1302', 'c1303'])
    print("\n无穷值统计:")
    print(np.isinf(df[['family_background', 'policy', 'courses']]).sum())
    print("\nNaN统计:")
    print(np.isnan(df[['family_background', 'policy', 'courses']]).sum())
    columns_to_check = ['correction', 'family_background', 'policy', 'courses', 'cog3pl']
    df = df.dropna(subset=columns_to_check, how='any')

    X = df[['correction', 'family_background', 'policy', 'courses']]
    X = sm.add_constant(X)
    y = df['cog3pl']
    model_c = sm.OLS(y, X).fit()
    print(model_c.summary())

    X = df[['family_background', 'policy', 'courses']]
    X = sm.add_constant(X)
    y = df['cog3pl']
    model_d = sm.OLS(y, X).fit()
    print(model_d.summary())
    

    """df = df.set_index(['ids', 'fall'])
    df['E'] = df['b14a1']
    # 创建滞后变量
    df['E_lag'] = df.groupby(level=0)['E'].shift()

    # 固定效应模型
    mod = PanelOLS.from_formula(
        "E ~ 1 + E_lag + EntityEffects",
        data=df.dropna())
    res = mod.fit()"""


    sns.kdeplot(x=df.G, y=df.M, cmap="viridis", fill=True)
    plt.title("G-M替代曲面")
    plt.xlabel("成绩产出G")
    plt.ylabel("心理健康M")
    plt.savefig('G_M_surface.png')

    # 社会资本互动效应
    sns.lmplot(x='S', y='G', hue='peer_E_quantile', 
            data=df.assign(peer_E_quantile=pd.qcut(df.peer_E, 3)))
    plt.title("社会资本S对成绩产出的调节效应")
    plt.savefig('S_G_interaction.png')