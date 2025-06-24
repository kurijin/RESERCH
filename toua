import pandas as pd
import numpy as np

def evaluate_estimate_vs_population(
    area_name,
    estimate_dir="/Users/kuritasachikawa/Desktop/KDDIデータ/評価/ISD国内",
    population_dir="/Users/kuritasachikawa/Desktop/KDDIデータ/国内集計",
    estimate_method="SVD",  # "SVD"や"NMF"など
    rank=3,
    estimate_prefix=None  # Noneなら area_name + _{method}_rank{rank}_ISD.csv
):
    # ファイルパス決定
    if estimate_prefix is None:
        estimate_file = f"{area_name}_{estimate_method}_rank{rank}_ISD.csv"
    else:
        estimate_file = estimate_prefix
    estimate_path = f"{estimate_dir}/{estimate_file}"
    population_path = f"{population_dir}/domestic_population_{area_name}.csv"

    # 推定値読み込み
    estimate_df = pd.read_csv(estimate_path)
    if 'date' not in estimate_df.columns:
        estimate_df.rename(columns={estimate_df.columns[0]: 'date'}, inplace=True)
    if isinstance(estimate_df['date'].iloc[0], str):
        estimate_df['date'] = pd.to_datetime(estimate_df['date']).dt.date
    # 横持ち形式なら melt
    if any(str(h).isdigit() for h in estimate_df.columns[1:]):
        estimate_df = estimate_df.melt(id_vars='date', var_name='hour', value_name='estimate')
        estimate_df['hour'] = estimate_df['hour'].astype(int)
    else:
        estimate_df['hour'] = estimate_df['hour'].astype(int)
    # 推定値がNaNなら0に
    estimate_df['estimate'] = estimate_df['estimate'].fillna(0)

    # 実測データ読み込み
    population_df = pd.read_csv(population_path)
    population_df['日付'] = pd.to_datetime(population_df['日付']).dt.date
    if population_df['時刻'].dtype == object:
        population_df['時刻'] = population_df['時刻'].str.replace("時", "").astype(int)
    true_grouped = population_df.groupby(['日付', '時刻'])['合計'].sum().reset_index()
    true_grouped.rename(columns={'日付': 'date', '時刻': 'hour', '合計': 'true'}, inplace=True)

    # 結合
    merged = pd.merge(estimate_df, true_grouped, on=['date', 'hour'], how='inner')
    valid = merged[merged['true'] > 0].copy()
    valid['squared_error'] = (valid['estimate'] - valid['true']) ** 2
    valid['abs_pct_error'] = np.abs(valid['estimate'] - valid['true']) / valid['true'] * 100

    # RMSE・MAPE
    rmse = np.sqrt(valid['squared_error'].mean())
    mape = valid['abs_pct_error'].mean()

    print(f"\n✅ {area_name} の {estimate_method} ランク{rank} 週合計→日別復元csv VS 人口実測")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")
    return 

# --- 使用例 ---
evaluate_estimate_vs_population("河原町エリア", estimate_method="ConstrainedNMF", rank=3)
evaluate_estimate_vs_population("清水寺エリア", estimate_method="ConstrainedNMF", rank=24)
evaluate_estimate_vs_population("嵯峨嵐山エリア", estimate_method="ConstrainedNMF", rank=24)
