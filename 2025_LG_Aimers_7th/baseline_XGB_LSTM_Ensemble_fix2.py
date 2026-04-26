## 

import os
import random
import glob
import re

import hashlib
import pandas as pd
import numpy as np

from tqdm import tqdm
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() :
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_seed(42)

LOOKBACK, PREDICT = 28, 7
CUTTOFF = pd.to_datetime("2024-06-15")

TRAIN_PATH = "data/train/train.csv"
TEST_DIR = "data/test"
SAMPLE_SUB_PATH = "data/sample_submission.csv"
OUTPUT_PATH = "submission.csv"

ENSEMBLE_MODE = "weighted"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _stable_bucket(text: str, mod: int = 50) -> int:
    return int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16) % mod

# XGBoost 피처
def calendar_features_for_date(d: pd.Timestamp, store_menu:str):
    day = d.weekday()
    month = d.month
    quarter = (month - 1) // 3 + 1
    week_of_month = (d.day - 1) // 7 + 1
    is_weekend = 1 if day >= 5 else 0

    day_sin = np.sin(2 * np.pi * day / 7.0)
    day_cos = np.cos(2 * np.pi * day / 7.0)
    mon_sin = np.sin(2 * np.pi * (month - 1) / 12.0)
    mon_cos = np.cos(2 * np.pi * (month - 1) / 12.0)

    store_name = store_menu.split("_")[0]
    menu_name = store_menu.split("_")[-1]

    store_bucket = _stable_bucket(store_name, 50)
    menu_bucket = _stable_bucket(menu_name, 50)
    
    inter_store_day = store_bucket * (day + 1)
    inter_menu_month = menu_bucket * month

    return np.array([
        day, month, quarter, week_of_month, is_weekend,
        day_sin, day_cos, mon_sin, mon_cos,
        inter_store_day, inter_menu_month
    ], dtype=np.float32)

# LSTM 피처
def _calendar_vec(d: pd.Timestamp):
    day = d.weekday()
    month = d.month
    quarter = (month - 1)//3 + 1
    week_of_month = (d.day - 1)//7 + 1
    is_weekend = 1 if day >= 5 else 0
    day_sin = np.sin(2*np.pi*day/7.0)
    day_cos = np.cos(2*np.pi*day/7.0)
    mon_sin = np.sin(2*np.pi*(month-1)/12.0)
    mon_cos = np.cos(2*np.pi*(month-1)/12.0)
    return np.array([day, month, quarter, week_of_month, is_weekend,
                     day_sin, day_cos, mon_sin, mon_cos], dtype=np.float32)

def _rolling_stats(series: np.ndarray, idx: int):
    end = idx
    start_28 = max(0, end-28)
    start_7 = max(0, end-7)
    w28 = series[start_28:end]
    w7 = series[start_7:end]
    mean28 = w28.mean() if len(w28)>0 else series[max(0,end-1)]
    mean7 = w7.mean() if len(w7)>0 else series[max(0,end-1)]
    lag7 = series[end-7] if end-7 >= 0 else series[max(0,end-1)]
    return np.array([lag7, mean7, mean28], dtype=np.float32)

def build_feature_matrix_for_group(g: pd.DataFrame):
    g = g.sort_values("영업일자").reset_index(drop=True)
    y = g["매출수량"].values.astype(np.float32)
    dates = pd.to_datetime(g["영업일자"]).values
    rows = []
    for i in range(len(g)):
        cal = _calendar_vec(pd.Timestamp(dates[i]))
        stats3 = _rolling_stats(y, i)
        row = np.concatenate(([y[i]], cal, stats3)).astype(np.float32)
        rows.append(row)
    X_full = np.stack(rows, axis=0)
    return X_full, y, dates

def build_lstm_sequences_multi(X_feat: np.ndarray, y: np.ndarray, LOOKBACK=28, PREDICT=7):
    X_list, Y_list = [], []
    for i in range(len(y) - LOOKBACK - PREDICT + 1):
        X_list.append(X_feat[i:i+LOOKBACK])
        Y_list.append(y[i+LOOKBACK:i+LOOKBACK+PREDICT])
    return np.array(X_list), np.array(Y_list)

def fit_scale_XY(X_seq: np.ndarray, Y_seq: np.ndarray):
    N, T, F = X_seq.shape
    x_scaler = MinMaxScaler()
    X2d = X_seq.reshape(-1, F)
    X2d_scaled = x_scaler.fit_transform(X2d).astype(np.float32)
    X_scaled = X2d_scaled.reshape(N, T, F)

    y_scaler = MinMaxScaler()
    Y2d = Y_seq.reshape(-1, 1)
    Y2d_scaled = y_scaler.fit_transform(Y2d).astype(np.float32)
    Y_scaled = Y2d_scaled.reshape(Y_seq.shape[0], Y_seq.shape[1])
    return X_scaled, Y_scaled, x_scaler, y_scaler
 
def scale_X_with(x_scaler: MinMaxScaler, X_seq: np.ndarray):
    N, T, F = X_seq.shape
    X2d = X_seq.reshape(-1, F)
    X2d_scaled = x_scaler.transform(X2d).astype(np.float32)
    return X2d_scaled.reshape(N, T, F)

def inverse_scale_y(y_scaler: MinMaxScaler, arr: np.ndarray):
    return y_scaler.inverse_transform(arr.reshape(-1, 1)).reshape(arr.shape)

def _get_best_n_from_model(model, default_n):
    try:
        return int(model.best_iteration + 1)
    except Exception:
        pass
    try:
        return int(model.get_booster().best_iteration + 1)
    except Exception:
        pass
    try:
        er = model.evals_result()
        hist = er.get('validation_0', {}).get('mae', None)
        if hist:
            return int(len(hist))
    except Exception:
        pass
    return int(default_n)


#data
train = pd.read_csv(TRAIN_PATH)
train["영업일자"] = pd.to_datetime(train["영업일자"])
train = train[train["영업일자"] <= CUTTOFF].copy()
train = train.sort_values(["영업장명_메뉴명", "영업일자"]).reset_index(drop=True)
train["매출수량"] = train["매출수량"].clip(lower=0)
print("학습 데이터 크기:", train.shape)


#윈도우생성
def build_training_windows(group_df: pd.DataFrame, store_menu: str):
    values = group_df["매출수량"].values
    dates = pd.to_datetime(group_df["영업일자"]).values

    X_dict = {h: [] for h in range(1, PREDICT+1)}
    Y_dict = {h: [] for h in range(1, PREDICT+1)}
    I_dict = {h: [] for h in range(1, PREDICT+1)}

    for i in range(LOOKBACK, len(values) - PREDICT + 1):
        window_vals = values[i-LOOKBACK:i]

        mean7 = window_vals[-7:].mean()
        mean28 = window_vals.mean()
        lag7 = window_vals[-7] if len(window_vals) >= 7 else window_vals[-1]

        for h in range(1, PREDICT+1):
            t_idx = i + (h - 1)
            target_val = values[t_idx]
            target_date = pd.Timestamp(dates[t_idx])

            cal_feats = calendar_features_for_date(target_date, store_menu)

            X = np.concatenate([
                window_vals.astype(np.float32),
                np.array([mean7, mean28, lag7], dtype=np.float32),
                cal_feats.astype(np.float32)
            ]).astype(np.float32)
            X_dict[h].append(X)
            Y_dict[h].append(target_val)
            I_dict[h].append(t_idx)

    for h in range(1, PREDICT+1):
        X_dict[h] = np.array(X_dict[h], dtype=float)
        Y_dict[h] = np.array(Y_dict[h], dtype=float)
        I_dict[h] = np.array(I_dict[h], dtype=int)
    return X_dict, Y_dict, I_dict

#cross validation, 최종학습
def train_xgb_direct_with_cv(train_df: pd.DataFrame, n_splits: int = 3):
    models = {}
    cv_scores = {}
    store_means = {}
    global_mean = train_df["매출수량"].mean()

    use_gpu = True
    base_params = dict(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=1.5,
        gamma=0.5,
        random_state=42,
        n_jobs=-1,
        eval_metric='mae',
        tree_method='hist',
        device='cuda' if use_gpu else 'cpu'
    )

    for store_menu, g in tqdm(train_df.groupby("영업장명_메뉴명"), desc="Training with CV (per store_menu)"):
        g = g.sort_values("영업일자")
        if len(g) < LOOKBACK + PREDICT + 10:
            continue

        X_dict, Y_dict, I_dict = build_training_windows(g, store_menu)
        models_h = {}
        cv_scores_h = {}
        trained_any = False

        for h in range(1, PREDICT+1):
            Xh, Yh, Ih = X_dict[h], Y_dict[h], I_dict[h]
            if len(Xh) < n_splits + 1:
                continue

            order = np.argsort(Ih)
            Xh, Yh = Xh[order], Yh[order]
            
            train_params ={
                "max_depth": base_params.get("max_depth", 6),
                "eta": base_params.get("learning_rate", 0.03),
                "subsample": base_params.get("subsample", 0.8),
                "colsample_bytree": base_params.get("colsample_bytree", 0.8),
                "min_child_weight": base_params.get("min_child_weight", 1),
                "alpha": base_params.get("reg_alpha", 0.0),
                "lambda": base_params.get("reg_lambda", 1.0),
                "gamma": base_params.get("gamma", 0.0),
                "objective": "reg:squarederror",
                "eval_metric": "mae",
                "tree_method": "gpu_hist" if torch.cuda.is_available() else "hist",
            }

            tscv = TimeSeriesSplit(n_splits=n_splits)
            fold_mae = []
            best_n_list = []

            for tr_idx, va_idx in tscv.split(Xh):
                X_tr, X_va = Xh[tr_idx], Xh[va_idx]
                Y_tr, Y_va = Yh[tr_idx], Yh[va_idx]

                dtrain = xgb.DMatrix(X_tr, label=Y_tr)
                dvalid = xgb.DMatrix(X_va, label=Y_va)

                booster = xgb.train(
                    params=train_params,
                    dtrain=dtrain,
                    num_boost_round=base_params["n_estimators"],
                    evals=[(dvalid, "validation_0")],
                    early_stopping_rounds=100,
                    verbose_eval=False,
                )

                if hasattr(booster, "best_iteration") and booster.best_iteration is not None:
                    best_n_fold = int(booster.best_iteration + 1)
                elif hasattr(booster, "best_ntree_limit") and booster.best_ntree_limit is not None:
                    best_n_fold = int(booster.best_ntree_limit)
                else:
                    best_n_fold = int(base_params["n_estimators"])

                if hasattr(booster, "best_ntree_limit") and booster.best_ntree_limit is not None:
                    pred = booster.predict(dvalid, ntree_limit=booster.best_ntree_limit)
                else:
                    pred = booster.predict(dvalid)

                fold_mae.append(mean_absolute_error(Y_va, pred))
                best_n_list.append(best_n_fold)
            
            mae_mean = float(np.mean(fold_mae))
            cv_scores_h[h] = {"mae_mean": mae_mean, "mae_fold": [float(m) for m in fold_mae]}

            best_n = int(np.median(best_n_list)) if best_n_list else int*base_params["n_estimators"]

            final_model = xgb.XGBRegressor(**base_params)
            final_model.set_params(n_estimators=best_n)
            final_model.fit(Xh, Yh)
            models_h[h] = final_model
            trained_any = True

        if trained_any:
            models[store_menu.strip()] = models_h
            cv_scores[store_menu.strip()] = cv_scores_h
            store_means[store_menu.strip()] = g["매출수량"].mean()

    return models, cv_scores, store_means, global_mean

#LSTM

class MultiOutputLSTM(nn.Module) :
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=7, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first = True, dropout=dropout if num_layers>1 else 0.0)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x) :
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]) 
    
def build_lstm_sequences(series_values: np.ndarray, LOOKBACK=28, PREDICT=7):
    X, Y = [], []
    for i in range(len(series_values) - LOOKBACK - PREDICT + 1):
        X.append(series_values[i:i+LOOKBACK])
        Y.append(series_values[i+LOOKBACK:i+LOOKBACK+PREDICT, 0])
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

def inverse_scale_array(arr: np.ndarray, scaler: MinMaxScaler):
    arr2 = arr.reshape(-1, 1)
    inv = scaler.inverse_transform(arr2).reshape(arr.shape)
    return inv

#Train LSTM
def train_lstm_with_cv(train_df: pd.DataFrame, n_splits: int=3,
                       epochs_final: int=50, epochs_cv: int=10,
                       batch_size: int=16, lr: float=1e-3):
    lstm_models = {}
    lstm_cv_scores = {}
    for store_menu, g in tqdm(train_df.groupby('영업장명_메뉴명'), desc = 'LSTM Training (per store_menu)'):
        g = g.sort_values("영업일자")
        if len(g) < LOOKBACK + PREDICT + 10:
            continue

        X_feat, y_raw, _ = build_feature_matrix_for_group(g)
        X_seq, Y_seq = build_lstm_sequences_multi(X_feat, y_raw, LOOKBACK, PREDICT)
        if len(X_seq) < n_splits + 1:
            continue

        X_scaled, Y_scaled, x_scaler, y_scaler = fit_scale_XY(X_seq, Y_seq)

        #CV
        tscv = TimeSeriesSplit(n_splits=n_splits)
        mae_by_h_folds = {h: [] for h in range(1, PREDICT+1)}
        for tr_idx, va_idx in tscv.split(X_scaled):
            X_tr, Y_tr = torch.tensor(X_scaled[tr_idx]).float().to(DEVICE), torch.tensor(Y_scaled[tr_idx]).float().to(DEVICE)
            X_va, Y_va =torch.tensor(X_scaled[va_idx]).float().to(DEVICE), torch.tensor(Y_scaled[va_idx]).float().to(DEVICE)

            model = MultiOutputLSTM(input_dim=X_tr.shape[-1], output_dim=PREDICT).to(DEVICE)
            optim = torch.optim.Adam(model.parameters(), lr=lr)
            crit = nn.SmoothL1Loss(beta=0.5)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs_cv)

            model.train()
            for ep in range(epochs_cv):
                perm = torch.randperm(X_tr.shape[0])
                for i in range (0, X_tr.shape[0], batch_size):
                    idx = perm[i:i+batch_size]
                    out = model(X_tr[idx])
                    loss = crit(out, Y_tr[idx])
                    optim.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optim.step()
                scheduler.step()
            model.eval()
            with torch.no_grad():
                pred_scaled = model(X_va).cpu().numpy()
                true_scaled = Y_va.cpu().numpy()
            pred = inverse_scale_y(y_scaler, pred_scaled.reshape(-1)).reshape(*pred_scaled.shape)
            true = inverse_scale_y(y_scaler, true_scaled.reshape(-1)).reshape(*true_scaled.shape)
            for h in range(PREDICT):
                mae = mean_absolute_error(true[:, h], pred[:, h])
                mae_by_h_folds[h+1].append(float(mae))

        lstm_cv_scores_h = {}
        for h in range(1, PREDICT+1):
            folds = mae_by_h_folds[h]
            if folds:
                lstm_cv_scores_h[h] = {"mae_mean": float(np.mean(folds)), "mae_fold": [float(m) for m in folds]}
        if not lstm_cv_scores_h:
            continue

        model_final = MultiOutputLSTM(input_dim=X_scaled.shape[-1], output_dim=PREDICT).to(DEVICE)
        optim = torch.optim.Adam(model_final.parameters(), lr=lr)
        crit = nn.SmoothL1Loss(beta=0.5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=epochs_final
        )
        X_full = torch.tensor(X_scaled).float().to(DEVICE)
        Y_full = torch.tensor(Y_scaled).float().to(DEVICE)

        model_final.train()
        for ep in range(epochs_final):
            perm = torch.randperm(X_full.shape[0])
            for i in range(0, X_full.shape[0], batch_size):
                idx = perm[i:i+batch_size]
                out = model_final(X_full[idx])
                loss = crit(out, Y_full[idx])
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model_final.parameters(), max_norm=1.0)
                optim.step()
            scheduler.step()

        k = store_menu.strip()
        lstm_models[k] = {"model": model_final.eval(), "x_scaler": x_scaler, "y_scaler": y_scaler}
        lstm_cv_scores[k] = lstm_cv_scores_h

    return lstm_models, lstm_cv_scores

  

xgb_models, xgb_cv_scores, store_means, global_mean = train_xgb_direct_with_cv(train, n_splits=3)
print("XGB 학습된 매장 메뉴 수:", len(xgb_models))

lstm_models, lstm_cv_scores = train_lstm_with_cv(train, n_splits=3, epochs_final=50, epochs_cv=10, batch_size=16, lr=1e-3)
print("LSTM 학습된 매장 메뉴 수:", len(lstm_models))

#예측
def predict_xgb_for_group(store_menu: str, last_window_vals: np.ndarray, last_date: pd.Timestamp):
    preds = []
    key = store_menu.strip()
    for h in range(1, PREDICT+1):
        cal = calendar_features_for_date(last_date + pd.Timedelta(days=h), store_menu)
        x = np.concatenate([
            last_window_vals.astype(np.float32),
            np.array([last_window_vals[-7:].mean(),
                      last_window_vals.mean(),
                      last_window_vals[-7] if len(last_window_vals) >= 7 else last_window_vals[-1]],
                      dtype=np.float32),
            cal.astype(np.float32)
        ], dtype=np.float32).reshape(1,-1)

        if (key in xgb_models) and (h in xgb_models[key]):
            y_pred = float(xgb_models[key][h].predict(x)[0])
        else:
            y_pred = float(store_means.get(key, global_mean))
        preds.append(max(y_pred, 0.0))
    return np.array(preds, dtype=float)

def predict_lstm_for_group(store_menu: str, recent_vals: np.ndarray, recent_dates: np.ndarray):
    key = store_menu.strip()
    if key not in lstm_models:
        return None
    model = lstm_models[key]["model"]
    x_scaler = lstm_models[key]["x_scaler"]
    y_scaler = lstm_models[key]["y_scaler"]

    rows = []
    y_series = recent_vals.astype(np.float32)
    for i in range(len(recent_vals)):
        cal = _calendar_vec(pd.Timestamp(recent_dates[i]))
        stats3 = _rolling_stats(y_series, i)
        row = np.concatenate(([y_series[i]], cal, stats3)).astype(np.float32)
        rows.append(row)
    X_last = np.stack(rows, axis=0)[None, ...]
    X_last_scaled = scale_X_with(x_scaler, X_last)

    with torch.no_grad():
        x_t = torch.tensor(X_last_scaled).float().to(DEVICE)
        pred_scaled = model(x_t).cpu().numpy().reshape(-1)
    pred = inverse_scale_y(y_scaler, pred_scaled)
    pred = np.clip(pred, 0, None)
    return pred

def get_weights_for_ensemble(store_menu: str, h: int):
    if ENSEMBLE_MODE == "mean":
        return 1.0, 1.0
    key = store_menu.strip()
    mae_x = None
    mae_l = None

    if key in xgb_cv_scores and h in xgb_cv_scores[key]:
        mae_x = xgb_cv_scores[key][h]["mae_mean"]

    if key in lstm_cv_scores and h in lstm_cv_scores[key]:
        mae_l = lstm_cv_scores[key][h]["mae_mean"]

    if (mae_x is not None) and (mae_l is not None):
        wx = 1.0 / (mae_x + 1e-6)
        wl = 1.0 / (mae_l + 1e-6)
        return wx, wl
    elif(mae_x is not None) and (mae_l is None):
        return 1.0, 0.0
    elif(mae_x is None) and (mae_l is not None):
        return 0.0, 1.0
    else:
        return 1.0, 1.0
    
def predict_ensemble_for_test_file(test_path: str):
    df = pd.read_csv(test_path)
    df["영업일자"] = pd.to_datetime(df["영업일자"])
    df = df.sort_values(["영업장명_메뉴명", "영업일자"])
    filename = os.path.basename(test_path)
    test_prefix = re.search(r"(TEST_\d+)", filename).group(1)

    preds_rows = []
    for store_menu, g in df.groupby("영업장명_메뉴명"):
        g = g.sort_values("영업일자")
        vals = g["매출수량"].values
        dates = pd.to_datetime(g["영업일자"].values)

        if len(vals) < LOOKBACK:
            base = store_means.get(store_menu.strip(), global_mean)
            for h in range(1, PREDICT+1):
                preds_rows.append({"영업일자": f"{test_prefix}+{h}일",
                                   "영업장명_메뉴명": store_menu,
                                   "매출수량": max(base, 0.0)})
            continue

        window = vals[-LOOKBACK:]
        last_date = pd.Timestamp(dates[-1])
        xgb_pred7 = predict_xgb_for_group(store_menu, window, last_date)
        dates_window = dates[-LOOKBACK:]
        lstm_pred7 = predict_lstm_for_group(store_menu, window, dates_window)

        for h in range(1, PREDICT+1):
            yx = xgb_pred7[h-1]
            if lstm_pred7 is None:
                y = yx
            else:
                yl = float(lstm_pred7[h-1])
                wx, wl = get_weights_for_ensemble(store_menu, h)
                if wx == 0 and wl ==0:
                    wx, wl = 1.0, 1.0
                y = (wx * yx + wl * yl) / (wx + wl)

            preds_rows.append({
                "영업일자": f"{test_prefix}+{h}일",
                "영업장명_메뉴명": store_menu,
                "매출수량": max(float(y), 0.0)
            })
            
    return pd.DataFrame(preds_rows)


test_files = sorted(glob.glob(os.path.join(TEST_DIR, "TEST_*.csv")))
all_preds = [predict_ensemble_for_test_file(p) for p in test_files]
full_pred_df = pd.concat(all_preds, ignore_index=True)

print("총 예측 수 :", len(full_pred_df))
print("0으로 예측된 수:", int((full_pred_df["매출수량"] == 0).sum()))

#포맷 변환
sample_submission = pd.read_csv(SAMPLE_SUB_PATH)

def to_submission(pred_df: pd.DataFrame, sample_df: pd.DataFrame):
    pred_dict = dict(zip(
        zip(pred_df["영업일자"].astype(str), pred_df["영업장명_메뉴명"].astype(str)),
        pred_df["매출수량"].astype(float)
    ))
    out = sample_df.copy()
    for i in out.index:
        tag = out.loc[i, "영업일자"]
        for col in out.columns[1:]:
            out.loc[i, col] = float(pred_dict.get((tag, col), 0.0))
    return out

submission = to_submission(full_pred_df, sample_submission)
for col in submission.columns[1:]:
    submission[col] = pd.to_numeric(submission[col], errors='coerce').fillna(0)
submission = submission.round(6)
submission.columns = sample_submission.columns
submission.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
print("저장완료:", OUTPUT_PATH)
