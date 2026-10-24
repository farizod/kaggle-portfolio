"""Train models and build submissions for 7 Kaggle competitions.

Usage: python scripts/model_all.py <comp_key>
comp_key one of: spaceship forest s4e1 s4e2 s4e10 house digit  (or 'all')
"""
import sys, os, warnings, time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
import lightgbm as lgb

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # kaggle_work (no '..', shorter path)
DATA = os.path.join(BASE, "data")
SUBS = os.path.join(BASE, "subs")
os.makedirs(SUBS, exist_ok=True)


def dpath(comp, f):
    return os.path.join(DATA, comp, f)


def to_category(train, test, cols):
    for c in cols:
        train[c] = train[c].astype("category")
        test[c] = test[c].astype("category")
        # unify categories
        cats = pd.api.types.union_categoricals([train[c], test[c]]).categories
        train[c] = pd.Categorical(train[c], categories=cats)
        test[c] = pd.Categorical(test[c], categories=cats)
    return train, test


# ---------------- generic trainers ----------------

def cv_binary_auc(X, y, Xtest, params, n_splits=5, seed=42):
    oof = np.zeros(len(X)); pred = np.zeros(len(Xtest))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in skf.split(X, y):
        m = lgb.LGBMClassifier(**params)
        m.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[va], y.iloc[va])],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        pred += m.predict_proba(Xtest)[:, 1] / n_splits
    return roc_auc_score(y, oof), pred


def cv_binary_acc(X, y, Xtest, params, n_splits=5, seed=42):
    oof = np.zeros(len(X)); pred = np.zeros(len(Xtest))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in skf.split(X, y):
        m = lgb.LGBMClassifier(**params)
        m.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[va], y.iloc[va])],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        pred += m.predict_proba(Xtest)[:, 1] / n_splits
    acc = accuracy_score(y, (oof > 0.5).astype(int))
    return acc, pred


def cv_multiclass(X, y, Xtest, params, n_splits=5, seed=42):
    nclass = y.nunique()
    oof = np.zeros((len(X), nclass)); pred = np.zeros((len(Xtest), nclass))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in skf.split(X, y):
        m = lgb.LGBMClassifier(**params)
        m.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[va], y.iloc[va])],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        oof[va] = m.predict_proba(X.iloc[va])
        pred += m.predict_proba(Xtest) / n_splits
    acc = accuracy_score(y, oof.argmax(1))
    return acc, pred


def cv_reg(X, y, Xtest, params, n_splits=5, seed=42):
    oof = np.zeros(len(X)); pred = np.zeros(len(Xtest))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in kf.split(X):
        m = lgb.LGBMRegressor(**params)
        m.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[va], y.iloc[va])],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        oof[va] = m.predict(X.iloc[va])
        pred += m.predict(Xtest) / n_splits
    rmse = np.sqrt(mean_squared_error(y, oof))
    return rmse, pred


CLF = dict(n_estimators=2000, learning_rate=0.03, num_leaves=63,
           subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
           reg_lambda=1.0, min_child_samples=30, random_state=42, n_jobs=-1, verbose=-1)
REG = dict(n_estimators=3000, learning_rate=0.02, num_leaves=31,
           subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
           reg_lambda=1.0, min_child_samples=20, random_state=42, n_jobs=-1, verbose=-1)


# ---------------- per-competition ----------------

def run_spaceship():
    comp = "spaceship-titanic"
    tr = pd.read_csv(dpath(comp, "train.csv"))
    te = pd.read_csv(dpath(comp, "test.csv"))
    ids = te["PassengerId"]
    y = tr["Transported"].astype(int)

    def fe(df):
        df = df.copy()
        g = df["PassengerId"].str.split("_", expand=True)
        df["Group"] = g[0].astype(int)
        df["PersonNum"] = g[1].astype(int)
        gs = df.groupby("Group")["PassengerId"].transform("count")
        df["GroupSize"] = gs
        cab = df["Cabin"].str.split("/", expand=True)
        df["Deck"] = cab[0]
        df["CabinNum"] = pd.to_numeric(cab[1], errors="coerce")
        df["Side"] = cab[2]
        spend = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
        df["TotalSpend"] = df[spend].sum(axis=1)
        df["NoSpend"] = (df["TotalSpend"] == 0).astype(int)
        for c in ["CryoSleep", "VIP"]:
            df[c] = df[c].astype("float")
        return df.drop(columns=["PassengerId", "Cabin", "Name"])

    trf, tef = fe(tr).drop(columns=["Transported"]), fe(te)
    cat = ["HomePlanet", "Destination", "Deck", "Side"]
    trf, tef = to_category(trf, tef, cat)
    acc, pred = cv_binary_acc(trf, y, tef, CLF)
    out = pd.DataFrame({"PassengerId": ids, "Transported": (pred > 0.5).astype(bool)})
    out.to_csv(os.path.join(SUBS, comp + ".csv"), index=False)
    return comp, "accuracy", acc


def run_forest():
    comp = "forest-cover-type-prediction"
    tr = pd.read_csv(dpath(comp, "train.csv"))
    te = pd.read_csv(dpath(comp, "test.csv"))
    ids = te["Id"]
    y = tr["Cover_Type"] - 1  # 0-indexed for lgbm
    X = tr.drop(columns=["Id", "Cover_Type"])
    Xt = te.drop(columns=["Id"])
    acc, pred = cv_multiclass(X, y, Xt, CLF)
    out = pd.DataFrame({"Id": ids, "Cover_Type": pred.argmax(1) + 1})
    out.to_csv(os.path.join(SUBS, comp + ".csv"), index=False)
    return comp, "accuracy", acc


def run_s4e1():
    comp = "playground-series-s4e1"
    tr = pd.read_csv(dpath(comp, "train.csv"))
    te = pd.read_csv(dpath(comp, "test.csv"))
    ids = te["id"]
    y = tr["Exited"]
    drop = ["id", "CustomerId", "Exited"]
    X = tr.drop(columns=drop); Xt = te.drop(columns=["id", "CustomerId"])
    cat = ["Geography", "Gender", "Surname"]
    X, Xt = to_category(X, Xt, cat)
    auc, pred = cv_binary_auc(X, y, Xt, CLF)
    out = pd.DataFrame({"id": ids, "Exited": pred})
    out.to_csv(os.path.join(SUBS, comp + ".csv"), index=False)
    return comp, "auc", auc


def run_s4e2():
    comp = "playground-series-s4e2"
    tr = pd.read_csv(dpath(comp, "train.csv"))
    te = pd.read_csv(dpath(comp, "test.csv"))
    ids = te["id"]
    ycat = tr["NObeyesdad"].astype("category")
    classes = list(ycat.cat.categories)
    y = ycat.cat.codes
    X = tr.drop(columns=["id", "NObeyesdad"]); Xt = te.drop(columns=["id"])
    cat = X.select_dtypes(exclude="number").columns.tolist()
    X, Xt = to_category(X, Xt, cat)
    acc, pred = cv_multiclass(X, y, Xt, CLF)
    labels = [classes[i] for i in pred.argmax(1)]
    out = pd.DataFrame({"id": ids, "NObeyesdad": labels})
    out.to_csv(os.path.join(SUBS, comp + ".csv"), index=False)
    return comp, "accuracy", acc


def run_s4e10():
    comp = "playground-series-s4e10"
    tr = pd.read_csv(dpath(comp, "train.csv"))
    te = pd.read_csv(dpath(comp, "test.csv"))
    ids = te["id"]
    y = tr["loan_status"]
    X = tr.drop(columns=["id", "loan_status"]); Xt = te.drop(columns=["id"])
    cat = X.select_dtypes(exclude="number").columns.tolist()
    X, Xt = to_category(X, Xt, cat)
    auc, pred = cv_binary_auc(X, y, Xt, CLF)
    out = pd.DataFrame({"id": ids, "loan_status": pred})
    out.to_csv(os.path.join(SUBS, comp + ".csv"), index=False)
    return comp, "auc", auc


def run_house():
    comp = "house-prices-advanced-regression-techniques"
    tr = pd.read_csv(dpath(comp, "train.csv"))
    te = pd.read_csv(dpath(comp, "test.csv"))
    ids = te["Id"]
    y = np.log1p(tr["SalePrice"])
    X = tr.drop(columns=["Id", "SalePrice"]); Xt = te.drop(columns=["Id"])
    obj = X.select_dtypes(exclude="number").columns.tolist()
    for c in obj:
        X[c] = X[c].fillna("None"); Xt[c] = Xt[c].fillna("None")
    num = [c for c in X.columns if c not in obj]
    for c in num:
        med = X[c].median()
        X[c] = X[c].fillna(med); Xt[c] = Xt[c].fillna(med)
    X, Xt = to_category(X, Xt, obj)
    rmse, pred = cv_reg(X, y, Xt, REG)
    out = pd.DataFrame({"Id": ids, "SalePrice": np.expm1(pred)})
    out.to_csv(os.path.join(SUBS, comp + ".csv"), index=False)
    return comp, "rmse_log", rmse


def run_digit():
    comp = "digit-recognizer"
    tr = pd.read_csv(dpath(comp, "train.csv"))
    te = pd.read_csv(dpath(comp, "test.csv"))
    y = tr["label"]
    X = tr.drop(columns=["label"]) / 255.0
    Xt = te / 255.0
    # single holdout with early stopping (faster than full 5-fold for 784 feats)
    from sklearn.model_selection import train_test_split
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
    params = dict(n_estimators=1500, learning_rate=0.05, num_leaves=127,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
                  reg_lambda=1.0, random_state=42, n_jobs=-1, verbose=-1)
    m = lgb.LGBMClassifier(**params)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    acc = accuracy_score(yva, m.predict(Xva))
    pred = m.predict(Xt)
    out = pd.DataFrame({"ImageId": np.arange(1, len(Xt) + 1), "Label": pred})
    out.to_csv(os.path.join(SUBS, comp + ".csv"), index=False)
    return comp, "accuracy", acc


RUNNERS = {
    "spaceship": run_spaceship, "forest": run_forest, "s4e1": run_s4e1,
    "s4e2": run_s4e2, "s4e10": run_s4e10, "house": run_house, "digit": run_digit,
}


if __name__ == "__main__":
    keys = sys.argv[1:] or ["all"]
    if keys == ["all"]:
        keys = list(RUNNERS.keys())
    for k in keys:
        t0 = time.time()
        comp, metric, score = RUNNERS[k]()
        print(f"[DONE] {k:10s} {comp:48s} {metric}={score:.5f}  ({time.time()-t0:.0f}s)", flush=True)
