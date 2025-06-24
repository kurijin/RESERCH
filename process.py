import os
import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.decomposition import NMF
from scipy.optimize import minimize

def window_sum_constraint_loss(H_flat, W, num_weeks, week_len, hour2window_sum, all_dates, all_hours, start_date, alpha=10.0):
    """ウィンドウごとの合計値制約付きNMFの損失関数"""
    H = H_flat.reshape((W.shape[1], len(all_hours)))
    X_recon = np.dot(W, H)
    # NMF再構成誤差
    recon_loss = np.linalg.norm(X_recon - X_recon, ord='fro') ** 2  # dummy, 0になる（W固定なので）
    # ウインドウ合計制約
    constraint_loss = 0
    for win_idx in range(num_weeks):
        win_start = pd.to_datetime(start_date) + timedelta(days=win_idx)
        win_dates = [win_start + timedelta(days=d) for d in range(week_len)]
        idxs = [all_dates.index(wd.date()) for wd in win_dates if wd.date() in all_dates]
        for hidx, hour in enumerate(all_hours):
            pred_sum = np.sum(X_recon[idxs, hidx])
            constraint_loss += (pred_sum - hour2window_sum[hour][win_idx]) ** 2
    total_loss = recon_loss + alpha * constraint_loss
    return total_loss

def perform_constrained_nmf_from_weekly_windows(
    area_name,
    isd_base_dir="/Users/kuritasachikawa/Desktop/KDDIデータ/ISD",
    output_dir="/Users/kuritasachikawa/Desktop/KDDIデータ/評価/ISD国内",
    rank=3,
    start_date="2024-01-01",  # データの初日
    week_len=7,               # 1ウィンドウあたりの日数
    alpha=10.0                # 制約の強さ
):
    # ISDファイルのパス取得
    isd_folder = os.path.join(isd_base_dir, area_name)
    isd_files = sorted(
        [f for f in os.listdir(isd_folder) if f.startswith("window_") and f.endswith(".csv")],
        key=lambda x: int(x.replace("window_", "").replace(".csv", ""))
    )

    # 日付リスト構築
    num_weeks = len(isd_files)
    all_dates = []
    for i in range(num_weeks):
        start = pd.to_datetime(start_date) + timedelta(days=i)
        for d in range(week_len):
            all_dates.append(start + timedelta(days=d))
    all_dates = sorted(set(all_dates))
    all_dates = [d.date() for d in all_dates]

    # 各時刻ごとに「A x = b」を作って最小二乗で推定
    all_hours = set()
    hour2window_sum = dict()
    for win_idx, file in enumerate(isd_files):
        df = pd.read_csv(os.path.join(isd_folder, file))
        if df["時刻"].dtype == object:
            df["時刻"] = df["時刻"].str.replace("時", "").astype(int)
        for _, row in df.iterrows():
            hour = row["時刻"]
            all_hours.add(hour)
            if hour not in hour2window_sum:
                hour2window_sum[hour] = []
            hour2window_sum[hour].append(row["合計"])

    all_hours = sorted(all_hours)
    date2idx = {d: i for i, d in enumerate(all_dates)}
    estimate_matrix = np.zeros((len(all_dates), len(all_hours)))
    for hidx, hour in enumerate(all_hours):
        # 行列A: (ウィンドウ数, 日数) 各ウィンドウがどの日付をカバーしているか
        A = []
        for i in range(num_weeks):
            win_start = pd.to_datetime(start_date) + timedelta(days=i)
            win_dates = [win_start + timedelta(days=d) for d in range(week_len)]
            row = [1 if d in [wd.date() for wd in win_dates] else 0 for d in all_dates]
            A.append(row)
        A = np.array(A)
        b = np.array(hour2window_sum[hour])  # shape=(num_weeks,)
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        estimate_matrix[:, hidx] = np.maximum(x, 0)

    # NMFによる初期分解
    nmf = NMF(n_components=rank, init='nndsvda', random_state=0, max_iter=10000)
    W = nmf.fit_transform(estimate_matrix)
    H = nmf.components_

    # Hのみ合計制約付きで再最適化
    res = minimize(
        window_sum_constraint_loss,
        H.flatten(),
        args=(W, num_weeks, week_len, hour2window_sum, all_dates, all_hours, start_date, alpha),
        method='L-BFGS-B',
        bounds=[(0, None)] * H.size,
        options={'maxiter': 1000}
    )
    H_opt = res.x.reshape(H.shape)
    approx = np.dot(W, H_opt)

    # DataFrame化
    result_records = []
    for i, d in enumerate(all_dates):
        for j, h in enumerate(all_hours):
            result_records.append({
                'date': d,
                'hour': h,
                'estimate': max(approx[i, j], 0)
            })
    estimate_df = pd.DataFrame(result_records)

    # 横持ちCSVで保存
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{area_name}_ConstrainedNMF_rank{rank}_ISD.csv")
    pivot_df = estimate_df.pivot(index="date", columns="hour", values="estimate")
    pivot_df.reset_index(inplace=True)
    pivot_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"→ {output_path} に保存しました")

# --- 使用例 ---
perform_constrained_nmf_from_weekly_windows("河原町エリア", rank=24, start_date="2024-01-01")
perform_constrained_nmf_from_weekly_windows("清水寺エリア", rank=24, start_date="2024-01-01")
perform_constrained_nmf_from_weekly_windows("嵯峨嵐山エリア", rank=24, start_date="2024-01-01")
