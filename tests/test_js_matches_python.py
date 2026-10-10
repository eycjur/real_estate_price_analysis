"""ブラウザ用の JS 実装(docs/js/)のテスト。

- 回帰(model.js)が、参照実装(reap.model)と同じ推定結果を出すこと
- 画面から呼ぶ分析処理(service.js)のテスト(tests/js/service.test.mjs)を、合成データの書き出し結果に対して実行
"""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from reap import export
from reap.model import Spec, age_effect, fit
from test_model import synthetic

ROOT = Path(__file__).resolve().parents[1]
VARS = ["structure", "renovated", "dup"]  # dup は structure と完全に共線(JS でも同じ列が落ちること)


def prepared(seed, city, ward_prefix=""):
    df = synthetic(n=3000, seed=seed)
    df["price"] = (df["price"] / 10000).round() * 10000  # 書き出し形式は1万円単位
    df["area"] = df["area"].round()
    df["ward"] = ward_prefix + df["ward"]
    df["district_key"] = df["ward"] + " " + df["district_key"].str.split(" ").str[1]
    return df.assign(city=city, kind="mansion", ward_code=lambda d: d["ward_code"] + (10 if ward_prefix else 0))


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    """合成データを本番と同じ形式(reap.export)で書き出したディレクトリ。"""
    out = tmp_path_factory.mktemp("site")
    tx = pd.concat([prepared(1, "X市"), prepared(2, "Y市", "Y")], ignore_index=True)
    levels = {v: list(tx[v].value_counts().index) for v in VARS}
    keys = tx["district_key"].drop_duplicates()
    geo = pd.DataFrame({"district_key": keys[~keys.str.startswith("C区")], "lat": 35.0, "lon": 139.0})  # C区は位置不明
    datasets = {}
    for i, (city, g) in enumerate(tx.groupby("city", sort=False)):
        info, blob = export.dataset(g, "mansion", levels, geo)
        (out / f"m{i}.json").write_text(json.dumps(info), encoding="utf-8")
        (out / f"m{i}.bin.gz").write_bytes(blob)
        datasets[city] = {"file": f"m{i}", "n": len(g), "wards": info["wards"]}
    wards = sorted(tx["ward"].unique())
    meta = {
        "cities": ["X市", "Y市"], "all_label": "全都市", "kinds": {"mansion": "中古マンション"},
        "datasets": {"mansion": datasets}, "levels": {"mansion": levels}, "labels": {v: v for v in VARS},
        "no_district": export.NO_DISTRICT, "district_lambda": 5.0, "year_min": 2015, "year_max": 2020,
        "n_total": len(tx), "population_last_actual_year": 2020,
        "variables": {"mansion": ["structure", "renovated"]},  # 画面側は共線な dup を使わない
        "reference": {"mansion": {"ref_age": 10, "ref_area": 60, "area_step": 5, "ref_station": 8, "station_step": 5, "base_levels": {}}},
        "population": {w: {"year": list(range(2010, 2051)), "population": [100000 + 500 * i * (j + 1) for i in range(41)]}
                       for j, w in enumerate(wards)},
        "rent": [{"ward": w, "area_min": 0, "area_max": 99, "rent": r, "n_units": 100}
                 for w, r in [("A区", 50000.0), ("B区", 60000.0), ("YA区", 40000.0)]],  # C区 などは家賃統計なし
        "rent_age": {"X市": {"age": [0.0, 40.0], "factor": [1.2, 0.8]}},
        "population_elasticity": {"mansion": {"coef": 0.5, "se": 0.1, "ci_low": 0.3, "ci_high": 0.7, "n": 1,
                                              "r2": 0.9, "se_type": "cluster(区)"}},
    }
    (out / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    # 金利と不動産価格指数(2015〜2020年、月次)。金利(短期プライムレートのみ)は年ごとに 1.0, 1.1, ... %、指数は 2010年=100 で毎月 +1
    months = [f"{y}-{m:02d}" for y in range(2015, 2021) for m in range(1, 13)]
    rates = {
        "from": "2015-01", "month": months, "labels": export.RATE_LABELS,
        "rates": {k: [1.0 + (int(mo[:4]) - 2015) / 10 if k == "prime_short" else None for mo in months] for k in export.RATE_LABELS},
        "rpi": {"全国": {c: [100 + i for i in range(len(months))] for c in ("total", "house", "mansion")}},
        "rpi_region": {"X市": "全国", "Y市": "全国", "全都市": "全国"},
    }
    (out / "rates.json").write_text(json.dumps(rates), encoding="utf-8")
    # 市況(2024〜2025年の月次)。首都圏: 成約100・新規登録400・在庫1200件/月、㎡単価は 成約50+i・新規60+i・在庫65+i(i=月の番号)
    # 城東地区: 成約50・新規100・在庫300。土地は首都圏だけで㎡単価なし(価格のみ)。価格帯は2区分×8四半期
    mo = [f"{y}-{m:02d}" for y in (2024, 2025) for m in range(1, 13)]
    st = lambda n, p=None: {"n": [n] * 24, **({"unit_price": [p + i for i in range(24)]} if p is not None else {}), "age": [20.0] * 24}  # noqa: E731
    market = {
        "kinds": {"mansion": "中古マンション", "land": "土地"}, "status": export.REINS_STATUS,
        "reins": {"month": mo, "regions": ["首都圏", "東京都 城東地区"], "region_notes": {"首都圏": "", "東京都 城東地区": "台東区、…"},
                  "series": {"mansion": {"首都圏": {"sold": st(100, 50), "new": st(400, 60), "stock": st(1200, 65)},
                                         "東京都 城東地区": {"sold": st(50, 80), "new": st(100, 90), "stock": st(300, 95)}},
                             "land": {"首都圏": {k: {"n": [n] * 24, "price": [3000.0] * 24, "age": [None] * 24}
                                                for k, n in [("sold", 10), ("new", 20), ("stock", 60)]}}},
                  "quarter": [f"{y}-Q{q}" for y in (2024, 2025) for q in range(1, 5)], "bands": {"mansion": ["~1000", "1000~"]},
                  "band_n": {"mansion": {"首都圏": {"sold": [[10] * 8, [20] * 8], "new": [[40] * 8, [40] * 8], "stock": [[100] * 8, [50] * 8]}}}},
        "sales_index": {"month": mo, "labels": export.SALES_INDEX_LABELS, "regions": ["全国"],
                        "series": {"全国": {**{k: [100.0] * 24 for k in export.SALES_INDEX_LABELS},
                                          **{f"{k}_n": [10] * 24 for k in export.SALES_INDEX_LABELS}}}},
        "starts": {"month": mo, "labels": export.STARTS_LABELS, "regions": ["全国"],
                   "series": {"全国": {"total": [30] * 24, "owner": [10] * 24, "rental": [10] * 24, "sale_mansion": [5] * 24,
                                     "sale_house": [5] * 24, "company": [0] * 24}}},
    }
    (out / "market.json").write_text(json.dumps(market), encoding="utf-8")
    # ローン: 機構の賃貸住宅融資は2020-07〜2021-06(金利の系列より後まである)、日銀の新規貸出は4四半期
    cm = [f"{y}-{m:02d}" for y, ms in ((2020, range(7, 13)), (2021, range(1, 7))) for m in ms]
    loan = {
        "chintai": {"month": cm, "labels": export.CHINTAI_LABELS, "rates": {k: [2.0] * 12 for k in export.CHINTAI_LABELS}},
        "boj_loans": {"quarter": ["2020-Q1", "2020-Q2", "2020-Q3", "2020-Q4"], "labels": export.BOJ_LOAN_LABELS,
                      "values": {"housing_new": [1, 2, 3, 4], "rental_new": [None, 1, 1, 1]}},
        "products": [{"category": "housing", "bank": "A銀行", "product": "変動", "rate_type": "変動", "rate_min": 0.5, "rate_max": None,
                      "condition": None, "as_of": "2021-06-01", "url": "https://example.com/"}],
        "rate_type_share": {"survey": ["第1回"], "variable_pct": [70.0], "fixed_period_pct": [20.0], "fixed_full_pct": [10.0], "note": [None]},
    }
    (out / "loan.json").write_text(json.dumps(loan), encoding="utf-8")
    # 人口: 年 2015/2020/2050 × 年齢(5歳階級の下限) 0/15/65/75。男女とも 2020年は 100/200/60/40 人(区)。
    # 区は2015年なし・2050年に半減、全国・都・23区は区の10倍で一定
    ages, years = [0, 15, 65, 75], [2015, 2020, 2050]
    base = [100, 200, 60, 40]
    ward = [None] * 4 + base + [v // 2 for v in base]
    big = [v * 10 for v in base] * 3
    pop = {"years": years, "ages": ages, "actual_last": 2020, "names": {"00000": "全国", "13000": "東京都", "13100": "東京23区", "13101": "千代田区"},
           "major": ["13100"], "prefs": ["13000"], "nation": "00000", "map_areas": ["13101"],
           "pop": {"00000": {"m": big, "f": big}, "13000": {"m": big, "f": big}, "13100": {"m": big, "f": big}, "13101": {"m": ward, "f": ward}}}
    (out / "population.json").write_text(json.dumps(pop, ensure_ascii=False), encoding="utf-8")
    (out / "boundaries.json").write_text(json.dumps({"13101": [[[139.7, 35.6], [139.8, 35.6], [139.8, 35.7]]]}), encoding="utf-8")
    # 空室率: 全国・都道府県・東京23区は2008年から、千代田区は2013年からの3時点(2018年は欠け)
    vac = {"years": [2008, 2013, 2018, 2023], "map_years": [2013, 2018, 2023], "sale_until": 1998, "nation": "00000", "major": ["13100"],
           "prefs": ["13000"], "map_areas": ["13101"],
           "names": {"00000": "全国", "13000": "東京都", "13100": "東京23区", "13101": "千代田区"},
           "rate": {"00000": [0.18, 0.19, 0.18, 0.2], "13000": [0.15, 0.16, 0.15, 0.15], "13100": [0.14, 0.16, 0.14, 0.14],
                    "13101": [None, 0.16, None, 0.12]},
           "rental_vacant": {c: [100, 100, 100, 100] for c in ["00000", "13000", "13100", "13101"]},
           "rented": {c: [500, 500, 500, 500] for c in ["00000", "13000", "13100", "13101"]}}
    (out / "vacancy.json").write_text(json.dumps(vac, ensure_ascii=False), encoding="utf-8")
    # 長期推移(2014〜2021年)。X市: 地価は2015年から毎年2倍(最新2021年=100)、家賃は毎年 +10、物価は2020年まで毎年 +5%(2021年は未公表)。
    # Y市は家賃がない。市街地価格指数(東京区部)は2016年から。金利は2020年の12か月
    yrs = list(range(2014, 2022))
    land_x = [None] + [100 / 2 ** (2021 - y) for y in yrs[1:]]
    long = {
        "years": yrs, "cities": ["X市", "Y市"], "land_last_year": 2021, "rent_survey_year": 2020,
        "land": {"X市": land_x, "Y市": land_x}, "land_n": {"X市": [0] + [10] * 7, "Y市": [0] + [10] * 7},
        "land_level": {"X市": 200000, "Y市": 100000},
        "rent": {"X市": [100 + 10 * i for i in range(8)], "Y市": [None] * 8}, "rent_level": {"X市": 60000},
        "nation": {"cpi": [100 * 1.05 ** i for i in range(7)] + [None], "jrei_six": [50.0] * 8,
                   "jrei_tokyo": [None, None] + [10.0 * (i + 1) for i in range(6)], "jrei_nation": [40.0] * 8},
        "rates": {"month": months[-12:], "labels": export.LONG_RATE_LABELS,
                  "values": {k: [1.5 if k == "prime_short" else None] * 12 for k in export.LONG_RATE_LABELS}},
    }
    (out / "long.json").write_text(json.dumps(long, ensure_ascii=False), encoding="utf-8")
    # 物件の評価: 地区の代表点(35.0, 139.0)の上と約1.1km北に A区 の住宅地(30万・60万円/㎡、駅640m)、A区 の商業地、B区 の住宅地。建築単価は2年分
    appraisal = {
        "koji_year": 2021, "wards": ["A区", "B区"], "uses": ["住宅地", "商業地"],
        "points": {"ward": [0, 0, 0, 1], "use": [0, 0, 1, 0], "lat": [35.0, 35.01, 35.0, 35.0], "lon": [139.0, 139.0, 139.0, 139.0],
                   "price": [300000, 600000, 1000000, 200000], "change": [1.0, 2.0, 3.0, 4.0], "address": ["A区1", "A区2", "A区3", "B区1"],
                   "station": ["駅"] * 4, "station_m": [640, 640, 100, 640], "far": [200, 200, 400, 200]},
        "building_cost": {"year": [2022, 2023], "wood": [180.0, 200.0], "src": [350.0, 360.0], "rc": [300.0, 314.3], "steel": [250.0, 280.0]},
    }
    (out / "appraisal.json").write_text(json.dumps(appraisal, ensure_ascii=False), encoding="utf-8")
    return tx, out


@pytest.fixture(scope="module")
def result(site):
    tx, out = site
    run = subprocess.run(["node", str(ROOT / "tests/js/fit_check.mjs"), str(out)], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    return tx, json.loads(run.stdout)


def test_service_js(site):
    run = subprocess.run(["node", "--test", str(ROOT / "tests/js/service.test.mjs")], capture_output=True, text=True,
                         env={**os.environ, "SITE_DIR": str(site[1])})
    assert run.returncode == 0, run.stdout[-4000:] + run.stderr[-2000:]


def spec(**kw):
    return Spec(categoricals=tuple(VARS), district_lambda=5.0, ref_age=10, ref_area=60, ref_station=8, **kw)


def compare(js, py):
    assert set(js["coef"]) == set(py.design.names)
    for name, b, se in zip(py.design.names, py.beta, py.se):
        assert js["coef"][name] == pytest.approx([b, se], rel=1e-6, abs=1e-9), name
    assert [js["r2"], js["adj_r2"], js["rmse"], js["mae"], js["district_df"]] == pytest.approx(
        [py.r2, py.adj_r2, py.rmse, py.mae, py.district_df], rel=1e-9)
    assert len(js["dropped"]) == len(py.design.dropped) == 1
    for key, g in py.district_effects.items():
        assert js["gamma"][key] == pytest.approx(g, rel=1e-6, abs=1e-10)


def test_city_fit_with_district_l2_and_hc1(result):
    tx, js = result
    compare(js["city"], fit(tx[tx["city"] == "X市"], spec()))


def test_city_fit_with_area_and_station_step(result):
    tx, js = result
    compare(js["city_area"], fit(tx[tx["city"] == "X市"], spec(area_step=5, station_step=5)))


def test_pooled_fit_with_cluster_se(result):
    tx, js = result
    compare(js["pooled"], fit(tx, spec(fixed_effects="ward+cityyear"), cluster=True))


def test_prediction_and_age_effect(result):
    tx, js = result
    f = fit(tx[tx["city"] == "X市"], spec())
    row = pd.DataFrame([{"age": 23.0, "area": 72.0, "station_min": 5.0, "ward": "B区", "year": 2018, "city": "X市",
                         "district_key": js["predict_district"], "structure": f.design.levels["structure"][0],
                         "renovated": f.design.levels["renovated"][0], "dup": f.design.levels["dup"][0]}])
    # JS 側は cats を水準番号で渡している(structure=1, renovated=0, dup=0)ので、同じ水準名に直す
    levels = {v: list(tx[v].value_counts().index) for v in VARS}
    row = row.assign(structure=levels["structure"][1], renovated=levels["renovated"][0], dup=levels["dup"][0])
    mu, se = f.predict(row)
    assert [js["predict"]["mu"], js["predict"]["se"]] == pytest.approx([mu[0], se[0]], rel=1e-6)
    fa = fit(tx[tx["city"] == "X市"], spec(area_step=5, station_step=5))
    mu, se = fa.predict(pd.concat([row, row.assign(area=500.0, station_min=90.0)]))  # 500㎡・90分 はデータにないので最も近い区分
    assert [v for p in js["predict_area"] for v in (p["mu"], p["se"])] == pytest.approx([mu[0], se[0], mu[1], se[1]], rel=1e-6)
    assert [js["age_effect"]["effect"], js["age_effect"]["se"]] == pytest.approx(age_effect(f, 12, 31), rel=1e-6)
    assert js["age_effect_far"]["effect"] == pytest.approx(age_effect(f, 10, 200)[0], rel=1e-6)  # 最も近い築年数で代用
