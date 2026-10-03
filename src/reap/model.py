"""ヘドニック回帰 (目的変数: ln 取引価格)。

区ごとに X'X を積み上げる正規方程式で解くので、全都市プール(約100万件)でも
計画行列全体をメモリに載せずに推定できる。

地区(町名)効果は数千〜1万水準あるため、ダミー列は作らず L2 罰則つきで吸収する:
  min ||y − Xβ − Dγ||² + λ||γ||²
D'D が対角なので、地区別の合計だけでシューア補行列を作れば厳密解になる。
λ は「地区の推定値が λ 件ぶん区平均に引き寄せられる」強さで、混合効果モデルのランダム効果に相当する。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# カテゴリ変数の表示名(列名 → ラベル)
CATEGORICALS = {
    "structure": "構造", "renovated": "改装", "layout": "間取り", "zoning": "用途地域", "far": "容積率",
    "future_use": "今後の利用目的", "source": "価格情報の種類", "quarter": "取引四半期",
    "land_shape": "土地の形状", "road_dir": "前面道路の方位", "road_type": "前面道路の種類",
    "road_width": "前面道路の幅員", "frontage": "間口", "region": "地域", "far_usage": "容積率の消化率",
}
FE_LABEL = {"ward": "区", "year": "取引年", "_cy": "都市×年"}
FE_PREFIX = tuple(f"{v}=" for v in FE_LABEL)
STATION_CAP = 30  # 駅徒歩の区間は この分数以上を1区間にまとめる(元データも30分以上は「30分〜60分」などの幅でしかない)


@dataclass(frozen=True)
class Spec:
    """説明変数の構成。築年数は1年刻みのカテゴリ変数(基準 = ref_age)で入れ、曲線の形を仮定しない。"""

    use_area: bool = True
    area_step: int | None = None  # 指定すると面積を step㎡ 刻みのカテゴリ変数(基準 = ref_area を含む区分)にする。None なら ln 面積
    use_land: bool = False  # 戸建用: ln 土地面積
    use_station: bool = True
    station_step: int | None = None  # 指定すると駅徒歩分を step分 刻みの区間(30分以上は1区間)にする。None なら1次の項
    use_population: bool = False
    categoricals: tuple[str, ...] = ("structure", "renovated")
    # "ward+year": 区FE+年FE(都市内分析用) / "ward+cityyear": 区FE+都市×年FE(全都市プール用)
    fixed_effects: str = "ward+year"
    district_lambda: float | None = None  # 地区効果の L2 罰則 λ (None なら地区効果なし)
    # 基準の物件。築年数は ref_age 年を基準水準にし、連続変数は ln(面積/ref_area)、(駅徒歩−ref_station) の形で
    # 入れるので、切片は「基準の物件」の ln価格になる(当てはまりや他の係数の意味は変わらない)
    ref_age: float = 0.0
    ref_area: float = 1.0
    ref_land: float = 1.0
    ref_station: float = 0.0
    base_levels: tuple[tuple[str, str], ...] = ()  # カテゴリ変数の基準水準を指定(既定は最頻の水準)


def city_year(df: pd.DataFrame) -> pd.Series:
    return df["city"] + "|" + df["year"].astype(int).astype(str)


@dataclass
class Design:
    spec: Spec
    names: list[str]
    labels: list[str]
    levels: dict[str, list] = field(default_factory=dict)  # カテゴリ変数の水準(先頭が基準)
    keep: np.ndarray | None = None  # 多重共線で落とした列を除く列番号
    dropped: list[str] = field(default_factory=list)

    def matrix(self, df: pd.DataFrame) -> np.ndarray:
        s = self.spec
        cols: list[np.ndarray] = []
        if s.fixed_effects == "ward+year":
            cols.append(np.ones(len(df)))
        if s.use_area and s.area_step is None:
            cols.append(np.log(df["area"].to_numpy(float) / s.ref_area))
        if s.use_land:
            cols.append(np.log(df["land_area"].to_numpy(float) / s.ref_land))
        if s.use_station and s.station_step is None:
            cols.append(df["station_min"].to_numpy(float) - s.ref_station)
        if s.use_population:
            cols.append(np.log(df["population"].to_numpy(float)))
        X = np.column_stack(cols) if cols else np.zeros((len(df), 0))
        # カテゴリ変数は水準コードから直接 one-hot を立てる(水準ごとの比較より大幅に速い)
        n_dummy = sum(len(lv) - 1 for lv in self.levels.values())
        D = np.zeros((len(df), n_dummy))
        rows, off = np.arange(len(df)), 0
        for var, levels in self.levels.items():
            v = (city_year(df) if var == "_cy" else self.age_level(df["age"]) if var == "age"
                 else self.area_level(df["area"]) if var == "area"
                 else self.station_level(df["station_min"]) if var == "station" else df[var])
            codes = pd.Index(levels).get_indexer(v)  # 水準にない値(基準として落とした都市×年など)は -1
            hit = codes >= 1
            D[rows[hit], off + codes[hit] - 1] = 1.0
            off += len(levels) - 1
        X = np.hstack([X, D])
        return X if self.keep is None else X[:, self.keep]


    def age_level(self, age) -> np.ndarray:
        """築年数を、推定に使った水準のうち最も近い年に丸める(データにない築年数や60年超の予測用)。"""
        return _nearest(self.levels["age"], age)

    def area_level(self, area) -> np.ndarray:
        """面積を区分の下限(step㎡ 刻み)にし、推定に使った区分のうち最も近いものに丸める。"""
        return _nearest(self.levels["area"], area_bin(area, self.spec.area_step))

    def station_level(self, minutes) -> np.ndarray:
        """駅徒歩分を区間の下限にし、推定に使った区間のうち最も近いものに丸める。"""
        return _nearest(self.levels["station"], station_bin(minutes, self.spec.station_step))


def area_bin(area, step: int) -> np.ndarray:
    """面積の区分の下限。例: step=5 なら 20〜24.9㎡ → 20。"""
    return (np.floor(np.asarray(area, dtype=float) / step) * step).astype(int)


def station_bin(minutes, step: int) -> np.ndarray:
    """駅徒歩の区間の下限。例: step=5 なら 10〜14分 → 10、30分以上 → 30。"""
    return np.minimum(np.floor(np.asarray(minutes, dtype=float) / step) * step, STATION_CAP).astype(int)


def _nearest(levels: list, x: np.ndarray) -> np.ndarray:
    lv = np.sort(np.array(levels, dtype=float))
    x = np.asarray(x, dtype=float)
    i = np.clip(np.searchsorted(lv, x), 1, len(lv) - 1)
    return np.where(x - lv[i - 1] <= lv[i] - x, lv[i - 1], lv[i]).astype(int)


def build_design(df: pd.DataFrame, spec: Spec) -> Design:
    names, labels = [], []
    if spec.fixed_effects == "ward+year":
        names.append("const"); labels.append("定数項")
    def centered(label: str, ref: float, log: bool = False) -> str:
        if log:
            return f"ln {label}" if ref == 1 else f"ln({label}/{ref:g})"
        return label if ref == 0 else f"({label}−{ref:g})"

    if spec.use_area and spec.area_step is None:
        names.append("ln_area"); labels.append(centered("建物面積㎡", spec.ref_area, log=True))
    if spec.use_land:
        names.append("ln_land"); labels.append(centered("土地面積㎡", spec.ref_land, log=True))
    if spec.use_station and spec.station_step is None:
        names.append("station_min"); labels.append(centered("駅徒歩分", spec.ref_station))
    if spec.use_population:
        names.append("ln_pop"); labels.append("ln 区人口")

    def by_freq(col):
        return list(df[col].value_counts().index)

    # 築年数: 1年刻み。基準は ref_age 年(データになければ最も多い築年数)
    ages = df["age"].round().astype(int)
    base_age = int(spec.ref_age) if (ages == int(spec.ref_age)).any() else int(ages.mode().iloc[0])
    levels: dict[str, list] = {"age": [base_age] + sorted(set(ages.unique().tolist()) - {base_age})}
    # 面積: step㎡ 刻み。基準は ref_area を含む区分(データになければ最も多い区分)
    if spec.use_area and spec.area_step is not None:
        bins = pd.Series(area_bin(df["area"], spec.area_step))
        ref_bin = int(area_bin(spec.ref_area, spec.area_step))
        base_bin = ref_bin if (bins == ref_bin).any() else int(bins.mode().iloc[0])
        levels["area"] = [base_bin] + sorted(set(bins.unique().tolist()) - {base_bin})
    # 駅徒歩: step分 刻みの区間。基準は ref_station を含む区間(データになければ最も多い区間)
    if spec.use_station and spec.station_step is not None:
        bins = pd.Series(station_bin(df["station_min"], spec.station_step))
        ref_bin = int(station_bin(spec.ref_station, spec.station_step))
        base_bin = ref_bin if (bins == ref_bin).any() else int(bins.mode().iloc[0])
        levels["station"] = [base_bin] + sorted(set(bins.unique().tolist()) - {base_bin})
    levels.update({c: by_freq(c) for c in spec.categoricals})
    for var, base in spec.base_levels:
        if var in levels and base in levels[var]:
            levels[var] = [base] + [lv for lv in levels[var] if lv != base]
    if spec.fixed_effects == "ward+year":
        levels["ward"] = by_freq("ward")
        levels["year"] = [int(y) for y in sorted(df["year"].unique(), reverse=True)]  # 最新年を基準に
    else:
        # 区ダミーは全水準(定数項なし)、都市×年は各都市の最新年を基準として落とす
        cy = sorted(city_year(df).unique())
        latest = df.groupby("city")["year"].max().astype(int)
        base = {f"{c}|{y}" for c, y in latest.items()}
        levels["ward"] = ["__none__"] + by_freq("ward")
        levels["_cy"] = ["__none__"] + [c for c in cy if c not in base]
    unit = {**CATEGORICALS, **FE_LABEL, "age": "築年数", "area": "建物面積", "station": "駅徒歩"}

    def shown(var, x):
        if var == "area":
            return f"{x}〜{x + spec.area_step}㎡"
        if var == "station":
            return f"{x}分以上" if x >= STATION_CAP else f"{x}〜{x + spec.station_step - 1}分"
        return x

    for var, lv in levels.items():
        for x in lv[1:]:
            names.append(f"{var}={x}")
            base_txt = "" if lv[0] == "__none__" else f" (基準: {shown(var, lv[0])})"
            labels.append(f"{unit.get(var, var)}: {shown(var, x)}{base_txt}")
    return Design(spec, names, labels, levels)


def independent_columns(A: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """X'X を先頭から順にコレスキー分解し、既出の列の線形結合になっている列を除いた列番号を返す。

    例: 戸建では「成約価格」「土地の形状=不明」「地域=不明」が完全に一致するため、後から出る列を落とす。
    """
    k = len(A)
    L = np.zeros((k, k))  # L[j, i]: 列 j の、採用済み i 番目の列に対する成分
    keep: list[int] = []
    for j in range(k):
        row = np.zeros(len(keep))
        for i, c in enumerate(keep):
            row[i] = (A[j, c] - row[:i] @ L[c, :i]) / L[c, i]
        d = A[j, j] - row @ row
        if d > tol * max(A[j, j], 1e-300):
            L[j, : len(keep)] = row
            L[j, len(keep)] = math.sqrt(d)
            keep.append(j)
    return np.array(keep)


@dataclass
class Fit:
    design: Design
    beta: np.ndarray
    cov: np.ndarray
    n: int
    r2: float
    adj_r2: float
    rmse: float  # ln価格の残差標準偏差
    mae: float
    se_type: str
    district_effects: pd.Series | None = None  # 地区キー → γ
    district_df: float = 0.0  # 地区効果の実効自由度 Σ n_d/(n_d+λ)

    @property
    def se(self) -> np.ndarray:
        return np.sqrt(np.diag(self.cov))

    def coef(self, name: str) -> tuple[float, float]:
        i = self.design.names.index(name)
        return float(self.beta[i]), float(self.se[i])

    def table(self, include_fe: bool = True) -> list[dict]:
        rows = []
        for nm, lb, b, s in zip(self.design.names, self.design.labels, self.beta, self.se):
            if not include_fe and nm.startswith(FE_PREFIX):
                continue
            t = b / s if s > 0 else float("nan")
            p = math.erfc(abs(t) / math.sqrt(2)) if s > 0 else float("nan")  # n が大きいので正規近似
            rows.append({"name": nm, "label": lb, "coef": float(b), "se": float(s), "t": float(t), "p": float(p),
                         "ci_low": float(b - 1.96 * s), "ci_high": float(b + 1.96 * s)})
        return rows

    def gamma(self, df: pd.DataFrame) -> np.ndarray:
        """各行の地区効果(推定時にない地区・地区未指定は 0 = 区の平均的な立地)。"""
        if self.district_effects is None or "district_key" not in df:
            return np.zeros(len(df))
        return self.district_effects.reindex(df["district_key"]).fillna(0.0).to_numpy()

    def predict(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """ln価格の予測値と、その平均の標準誤差(地区効果の推定誤差は含まない)。"""
        X = self.design.matrix(df)
        return X @ self.beta + self.gamma(df), np.sqrt(np.einsum("ij,jk,ik->i", X, self.cov, X))


def fit(df: pd.DataFrame, spec: Spec, cluster: bool = False) -> Fit:
    """OLS(+地区効果の L2 吸収)。標準誤差は cluster=True なら区クラスタ頑健、そうでなければ HC1。"""
    design = build_design(df, spec)
    lam = spec.district_lambda
    groups = [g for _, g in df.groupby("ward_code", sort=False)]
    k = len(design.names)
    xtx, xty = np.zeros((k, k)), np.zeros(k)
    sx, sy, nd = [], [], []  # 地区別の Σx, Σy, 件数
    for g in groups:
        X = design.matrix(g)
        y = np.log(g["price"].to_numpy(float))
        xtx += X.T @ X
        xty += X.T @ y
        if lam is not None:
            t = pd.DataFrame(X).assign(_y=y, _n=1.0).groupby(g["district_key"].to_numpy()).sum()
            sx.append(t.drop(columns=["_y", "_n"])); sy.append(t["_y"]); nd.append(t["_n"])
    if lam is not None:
        sx, sy, nd = pd.concat(sx), pd.concat(sy), pd.concat(nd)
        w = 1.0 / (nd.to_numpy() + lam)
        Sx = sx.to_numpy()
        xtx = xtx - (Sx * w[:, None]).T @ Sx  # シューア補行列
        xty = xty - Sx.T @ (sy.to_numpy() * w)

    keep = independent_columns(xtx)
    if len(keep) < k:
        kept = set(keep.tolist())
        design.dropped = [design.labels[i] for i in range(k) if i not in kept]
        design.names = [design.names[i] for i in keep]
        design.labels = [design.labels[i] for i in keep]
        design.keep = keep
        xtx, xty, k = xtx[np.ix_(keep, keep)], xty[keep], len(keep)
    beta = np.linalg.solve(xtx, xty)
    bread = np.linalg.inv(xtx)

    gamma, W, ddf = None, None, 0.0
    if lam is not None:
        Sx = Sx[:, keep]
        gamma = pd.Series((sy.to_numpy() - Sx @ beta) * w, index=sx.index)
        W = pd.DataFrame(Sx * w[:, None], index=sx.index)  # 地区効果を払い出した X̃ = X − W[地区]
        ddf = float((nd.to_numpy() * w).sum())

    meat, sse, sae, sy1, sy2 = np.zeros((k, k)), 0.0, 0.0, 0.0, 0.0
    for g in groups:
        X = design.matrix(g)
        y = np.log(g["price"].to_numpy(float))
        e = y - X @ beta
        if lam is not None:
            key = g["district_key"].to_numpy()
            e -= gamma.reindex(key).to_numpy()
            X = X - W.reindex(key).to_numpy()
        sse += float(e @ e); sae += float(np.abs(e).sum()); sy1 += float(y.sum()); sy2 += float(y @ y)
        if cluster:
            s = X.T @ e
            meat += np.outer(s, s)
        else:
            Xe = X * e[:, None]
            meat += Xe.T @ Xe
    n, G = len(df), len(groups)
    p = k + ddf  # 実効パラメータ数
    adj = (G / (G - 1)) * ((n - 1) / (n - p)) if cluster else n / (n - p)
    cov = adj * bread @ meat @ bread
    r2 = 1 - sse / (sy2 - sy1**2 / n)
    return Fit(design, beta, cov, n, r2, 1 - (1 - r2) * (n - 1) / (n - p), math.sqrt(sse / (n - p)), sae / n,
               "cluster(区)" if cluster else "HC1", gamma, ddf)


def age_effect(f: Fit, a0: float, a1: float) -> tuple[float, float]:
    """築年数 a0→a1 による ln価格の変化とその標準誤差(築年数ダミーの係数の差)。"""
    names = f.design.names
    c = np.zeros(len(names))
    for a, sign in ((a1, 1.0), (a0, -1.0)):
        name = f"age={f.design.age_level([a])[0]}"
        if name in names:  # 基準の築年数は係数0
            c[names.index(name)] += sign
    return float(c @ f.beta), float(np.sqrt(c @ f.cov @ c))
