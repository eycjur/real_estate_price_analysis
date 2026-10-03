import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from reap.model import Spec, age_effect, build_design, fit


def synthetic(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    wards = rng.choice(["A区", "B区", "C区"], n, p=[0.5, 0.3, 0.2])
    df = pd.DataFrame({
        "ward": wards,
        "ward_code": pd.Series(wards).map({"A区": 1, "B区": 2, "C区": 3}).to_numpy(),
        "city": "X市",
        "year": rng.integers(2015, 2021, n),
        "age": rng.integers(0, 50, n).astype(float),
        "area": rng.uniform(20, 100, n),
        "station_min": rng.integers(1, 20, n).astype(float),
        "structure": rng.choice(["RC", "SRC"], n),
        "renovated": rng.choice(["未改装", "改装済み", "不明"], n),
    })
    df["district_key"] = df["ward"] + " " + rng.choice(list("abcdefgh"), n)
    df["dup"] = df["structure"].map({"RC": "x", "SRC": "y"})  # structure と完全に共線な変数
    df["population"] = 100000 + 1000 * (df["year"] - 2015) * df["ward_code"]
    district_effect = df["district_key"].map({k: v for k, v in zip(sorted(df["district_key"].unique()), rng.normal(0, 0.15, 24))})
    lnp = (15 + district_effect - 0.02 * df["age"] + 0.9 * np.log(df["area"]) - 0.01 * df["station_min"]
           + 0.1 * (df["ward"] == "B区") + 0.03 * (df["year"] - 2015) + rng.normal(0, 0.2, n))
    df["price"] = np.exp(lnp)
    return df


def test_matches_statsmodels_hc1():
    df = synthetic()
    spec = Spec()
    f = fit(df, spec)
    design = build_design(df, spec)
    ref = sm.OLS(np.log(df["price"].to_numpy()), design.matrix(df)).fit(cov_type="HC1")
    np.testing.assert_allclose(f.beta, ref.params, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(f.se, ref.bse, rtol=1e-6)
    assert abs(f.r2 - ref.rsquared) < 1e-9


def test_matches_statsmodels_cluster_pooled():
    df = pd.concat([synthetic(seed=1), synthetic(seed=2).assign(
        city="Y市", ward=lambda d: "Y" + d["ward"], ward_code=lambda d: d["ward_code"] + 10)])
    spec = Spec(use_population=True, fixed_effects="ward+cityyear")
    f = fit(df, spec, cluster=True)
    design = build_design(df, spec)
    ref = sm.OLS(np.log(df["price"].to_numpy()), design.matrix(df)).fit(
        cov_type="cluster", cov_kwds={"groups": df["ward_code"].to_numpy()})
    np.testing.assert_allclose(f.beta, ref.params, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(f.coef("ln_pop")[1], ref.bse[design.names.index("ln_pop")], rtol=1e-5)


def test_recovers_age_coefficient_and_age_effect():
    f = fit(synthetic(n=20000), Spec(ref_age=10))
    effect, se = age_effect(f, 10, 20)
    assert abs(effect - (-0.2)) < 3 * se  # 真値は 1年あたり −0.02
    assert np.isclose(effect, f.coef("age=20")[0])  # 基準(築10年)の係数は0
    assert np.isclose(age_effect(f, 20, 30)[0], f.coef("age=30")[0] - f.coef("age=20")[0])
    assert age_effect(f, 10, 80)[0] == age_effect(f, 10, 49)[0]  # データにない築年数は最も近い年で代用


def test_district_l2_matches_dense_ridge():
    df = synthetic(n=1500)
    lam = 5.0
    f = fit(df, Spec(district_lambda=lam))
    X = f.design.matrix(df)
    D = (df["district_key"].to_numpy()[:, None] == f.district_effects.index.to_numpy()[None, :]).astype(float)
    Z = np.hstack([X, D])
    P = np.diag([0.0] * X.shape[1] + [lam] * D.shape[1])  # 地区ダミーだけに罰則
    theta = np.linalg.solve(Z.T @ Z + P, Z.T @ np.log(df["price"].to_numpy()))
    np.testing.assert_allclose(f.beta, theta[: X.shape[1]], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(f.district_effects.to_numpy(), theta[X.shape[1]:], rtol=1e-6, atol=1e-8)
    assert f.rmse < fit(df, Spec()).rmse  # 地区差のあるデータでは誤差が減る
    unseen = df.head(3).assign(district_key="どこにもない地区")
    np.testing.assert_allclose(f.predict(unseen)[0], f.design.matrix(unseen) @ f.beta)


def test_perfectly_collinear_columns_are_dropped():
    f = fit(synthetic(), Spec(categoricals=("structure", "dup")))
    assert len(f.design.dropped) == 1 and f.design.dropped[0].startswith("dup")
    assert len(f.beta) == len(f.design.names) == f.design.matrix(synthetic().head(5)).shape[1]
    assert np.isfinite(f.se).all()


def test_centering_moves_only_the_intercept():
    df = synthetic()
    f0, f1 = fit(df, Spec()), fit(df, Spec(ref_age=20, ref_area=60, ref_station=8))
    assert f1.rmse == pytest.approx(f0.rmse) and f1.coef("ln_area")[0] == pytest.approx(f0.coef("ln_area")[0])
    np.testing.assert_allclose(f1.predict(df.head(50))[0], f0.predict(df.head(50))[0])
    assert age_effect(f1, 10, 30)[0] == pytest.approx(age_effect(f0, 10, 30)[0])
    ref = df.head(1).assign(age=20.0, area=60.0, station_min=8.0, structure=f1.design.levels["structure"][0],
                            renovated=f1.design.levels["renovated"][0], ward=f1.design.levels["ward"][0],
                            year=f1.design.levels["year"][0])
    assert f1.coef("const")[0] == pytest.approx(f1.predict(ref)[0][0])  # 切片 = 基準物件の ln価格


def test_area_step_dummies():
    df = synthetic(n=20000)
    spec = Spec(area_step=5, ref_area=60)
    f = fit(df, spec)
    design = build_design(df, spec)
    assert "ln_area" not in f.design.names and f.design.levels["area"][0] == 60  # 基準は 60〜65㎡
    assert f.design.labels[f.design.names.index("area=20")] == "建物面積: 20〜25㎡ (基準: 60〜65㎡)"
    ref = sm.OLS(np.log(df["price"].to_numpy()), design.matrix(df)).fit(cov_type="HC1")
    np.testing.assert_allclose(f.beta, ref.params, rtol=1e-6, atol=1e-8)
    # 真値は 0.9·ln(面積)。区分の中央どうしの比で比べる
    b, se = f.coef("area=30")
    assert abs(b - 0.9 * np.log(32.5 / 62.5)) < 4 * se
    # 区分の中はどの面積でも同じ予測、データにない面積は最も近い区分で代用
    rows = df.head(1).assign(area=[60.0]).loc[[df.index[0]] * 4].assign(area=[60.0, 64.9, 500.0, 99.0])
    mu = f.predict(rows)[0]
    assert mu[0] == pytest.approx(mu[1]) and mu[2] == pytest.approx(mu[3])


def test_station_step_blocks():
    df = synthetic(n=20000)
    df.loc[df.index[:200], "station_min"] = 45.0  # 30分以上は1区間
    spec = Spec(station_step=5, ref_station=10)
    f = fit(df, spec)
    assert "station_min" not in f.design.names and f.design.levels["station"] == [10, 0, 5, 15, 30]
    assert f.design.labels[f.design.names.index("station=30")] == "駅徒歩: 30分以上 (基準: 10〜14分)"
    ref = sm.OLS(np.log(df["price"].to_numpy()), build_design(df, spec).matrix(df)).fit(cov_type="HC1")
    np.testing.assert_allclose(f.beta, ref.params, rtol=1e-6, atol=1e-8)
    b, se = f.coef("station=15")  # 真値は 1分あたり −0.01。区間の平均(17 と 12分)の差
    assert abs(b - (-0.01 * 5)) < 4 * se
    rows = df.head(1).loc[[df.index[0]] * 3].assign(station_min=[30.0, 120.0, 22.0])  # 20〜24分はデータにないので最も近い区間
    mu = f.predict(rows)[0]
    assert mu[0] == pytest.approx(mu[1]) and mu[2] == pytest.approx(f.predict(rows.assign(station_min=15.0))[0][2])
