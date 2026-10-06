"""data/processed の parquet から、ブラウザ用の静的データを docs/data/ に書き出す。

回帰などの計算はすべてブラウザ側の JS (docs/js/) で行う。ここで事前計算するのは、
利用者の入力に依存しない「人口弾力性」(全都市プール推定) だけ。

使い方: uv run python -m reap.export [--base-url https://<user>.github.io/<repo>/]
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .etl import VACANCY_SALE_UNTIL, VACANCY_YEARS
from .model import CATEGORICALS, Spec, fit

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
SITE = ROOT / "docs"

ALL = "全都市"
KINDS = {"mansion": "中古マンション", "house": "中古戸建(土地と建物)", "bldg_rc": "一棟マンション(RC・SRC)", "bldg_wood": "一棟アパート(木造・鉄骨)"}
LAND_KINDS = ("house", "bldg_rc", "bldg_wood")  # 土地面積も使う種別
BLDG_VARIABLES = ["structure", "far_usage", "zoning", "far", "future_use", "quarter", "land_shape", "road_dir", "road_type",
                  "road_width", "frontage", "region"]
# 種別ごとの説明変数(すべて投入する)。JS もこの定義を meta.json 経由で使う
VARIABLES = {
    "mansion": ["structure", "renovated", "layout", "zoning", "far", "future_use", "source", "quarter"],
    "house": ["structure", "zoning", "far", "future_use", "source", "quarter", "land_shape", "road_dir", "road_type",
              "road_width", "frontage", "region"],
    # 一棟ものは取引価格情報(アンケート)だけで成約価格情報にはないため「価格情報の種類」は使わず、容積率の消化率を足す
    "bldg_rc": BLDG_VARIABLES, "bldg_wood": BLDG_VARIABLES,
}
# 切片に対応する「基準の物件」。連続変数はこの値からの差で入れ、カテゴリ項目は最頻の水準を基準にする
# (マンションは 20㎡ に合わせて間取りの基準を 1K に、価格情報の種類は成約価格に固定)。area_step があれば面積を その㎡刻みの区分で、
# station_step があれば駅徒歩を その分刻みの区間(30分以上は1区間)で入れる
REFERENCE = {
    "mansion": {"ref_age": 20, "ref_area": 20, "area_step": 5, "ref_station": 10, "station_step": 5,
                "base_levels": {"layout": "1K", "source": "成約価格(レインズ)"}},
    "house": {"ref_age": 20, "ref_area": 100, "area_step": 5, "ref_land": 100, "ref_station": 10, "station_step": 5,
              "base_levels": {"source": "成約価格(レインズ)"}},
    # 一棟ものの基準は取引の中央値付近の規模。面積の幅(30〜3000㎡)が大きく区分が多くなりすぎるため ln 面積で入れる
    "bldg_rc": {"ref_age": 20, "ref_area": 600, "ref_land": 300, "ref_station": 10, "station_step": 5, "base_levels": {}},
    "bldg_wood": {"ref_age": 20, "ref_area": 200, "ref_land": 180, "ref_station": 10, "station_step": 5, "base_levels": {}},
}
# 「金利と価格指数」タブ。金利の系列名と、都市に対応する不動産価格指数の地域
RATE_LABELS = {
    "policy_rate": "基準貸付利率(政策金利の上限)", "call_rate": "無担保コールレート O/N(月平均)",
    "prime_short": "短期プライムレート(最頻値)", "prime_long": "長期プライムレート",
    "jgb10": "10年国債利回り(月平均)",
    "lend_new_long": "貸出約定平均金利 新規/長期", "lend_new_short": "貸出約定平均金利 新規/短期",
    "lend_stock_long": "貸出約定平均金利 ストック/長期", "lend_stock_short": "貸出約定平均金利 ストック/短期",
}
RPI_REGION = {
    "東京23区": "東京都", "横浜市": "南関東圏", "川崎市": "南関東圏", "相模原市": "南関東圏", "さいたま市": "南関東圏", "千葉市": "南関東圏",
    "名古屋市": "愛知県", "大阪市": "大阪府", "堺市": "大阪府", "京都市": "京阪神圏", "神戸市": "京阪神圏",
    "札幌市": "北海道地方", "仙台市": "東北地方", "広島市": "中国地方", "北九州市": "九州・沖縄地方", "福岡市": "九州・沖縄地方",
}
RPI_ALL = "全国"  # 全都市プールに対応する地域
RATES_FROM = "2005-01"  # 画面に出す月次系列の開始月
DISTRICT_LAMBDA = 10.0  # 地区効果の L2 罰則。検証用データの誤差は λ=1〜30 でほぼ同じ
NO_DISTRICT = 0xFFFF


def _codes(values: pd.Series, levels: list) -> np.ndarray:
    return pd.Index(levels).get_indexer(values)


def dataset(g: pd.DataFrame, kind: str, levels: dict[str, list], geo: pd.DataFrame) -> tuple[dict, bytes]:
    """1都市×1種別の取引を、列指向のバイナリ(各列は4バイト境界に揃える)とその説明(JSON)にする。"""
    wards = list(g["ward"].value_counts().index)
    dk = g["district_key"].dropna()
    districts = list(dk.value_counts().index)
    cols: list[tuple[str, np.ndarray]] = [
        ("price", (g["price"] / 10000).round().astype("uint32").to_numpy()),  # 万円(元データが1万円単位)
        ("area", g["area"].astype("uint16").to_numpy()),
        ("age", g["age"].astype("uint8").to_numpy()),
        ("station_min", g["station_min"].astype("uint8").to_numpy()),
        ("year", (g["year"] - 2000).astype("uint8").to_numpy()),
        ("ward", _codes(g["ward"], wards).astype("uint8")),
        ("district", np.where(g["district_key"].isna(), NO_DISTRICT, _codes(g["district_key"], districts)).astype("uint16")),
    ]
    if kind in LAND_KINDS:
        cols.append(("land_area", g["land_area"].astype("uint16").to_numpy()))
    for c in levels:  # カテゴリ変数(水準番号)
        cols.append((c, _codes(g[c], levels[c]).astype("uint8")))
    buf, spec = bytearray(), []
    for name, arr in cols:
        buf.extend(b"\0" * (-len(buf) % 4))
        spec.append({"name": name, "dtype": str(arr.dtype), "offset": len(buf)})
        buf.extend(arr.tobytes())
    geo = geo.set_index("district_key").reindex(districts)
    ward_of = g.dropna(subset=["district_key"]).drop_duplicates("district_key").set_index("district_key")["ward"]
    info = {
        "n": len(g), "columns": spec, "wards": wards,
        "districts": [{"name": k.split(" ", 1)[1], "ward": wards.index(ward_of[k]),
                       "lat": None if np.isnan(geo.at[k, "lat"]) else round(float(geo.at[k, "lat"]), 5),
                       "lon": None if np.isnan(geo.at[k, "lon"]) else round(float(geo.at[k, "lon"]), 5)}
                      for k in districts],
    }
    return info, gzip.compress(bytes(buf), mtime=0)


def population_elasticity(tx: pd.DataFrame, kind: str) -> dict:
    """全都市プール・区FE+都市×年FEで ln(区人口) の係数を推定する(標準誤差は区クラスタ)。"""
    ref = REFERENCE[kind]
    spec = Spec(use_land=kind in LAND_KINDS, use_population=True, categoricals=tuple(VARIABLES[kind]),
                fixed_effects="ward+cityyear", ref_age=ref["ref_age"], ref_area=ref["ref_area"], area_step=ref.get("area_step"),
                station_step=ref.get("station_step"),
                ref_land=ref.get("ref_land", 1.0), ref_station=ref["ref_station"],
                base_levels=tuple(ref["base_levels"].items()))
    f = fit(tx[tx["kind"] == kind], spec, cluster=True)
    b, se = f.coef("ln_pop")
    return {"coef": b, "se": se, "ci_low": b - 1.96 * se, "ci_high": b + 1.96 * se, "n": f.n, "r2": f.r2,
            "se_type": f.se_type}


def rates_json() -> dict:
    """金利(月次)と不動産価格指数(地域別・月次)を、共通の月の並びに揃えて1つの JSON にする。"""
    rates = pd.read_parquet(PROCESSED / "rates.parquet")
    rpi = pd.read_parquet(PROCESSED / "rpi.parquet")
    rates = rates[rates["month"] >= RATES_FROM].set_index("month")
    months = list(rates.index)
    vals = lambda s: [None if pd.isna(v) else round(float(v), 4) for v in s.reindex(months)]  # noqa: E731
    regions = [RPI_ALL, *dict.fromkeys(RPI_REGION.values())]
    by_region = {r: g.set_index("month") for r, g in rpi.groupby("region")}
    missing = [r for r in regions if r not in by_region]
    if missing:
        raise SystemExit(f"不動産価格指数に地域がありません: {missing}")
    return {
        "from": RATES_FROM, "month": months, "labels": RATE_LABELS,
        "rates": {k: vals(rates[k]) for k in RATE_LABELS},
        "rpi": {r: {c: vals(by_region[r][c]) for c in ("total", "house", "mansion")} for r in regions},
        "rpi_region": {**RPI_REGION, ALL: RPI_ALL},
    }


# 「市況」タブ
MARKET_KINDS = {"mansion": "中古マンション", "house": "中古戸建", "new_house": "新築戸建", "land": "土地(100〜200㎡)"}
REINS_STATUS = {"sold": "成約", "new": "新規登録", "stock": "在庫"}
SALES_INDEX_LABELS = {"total": "合計", "house": "戸建住宅", "mansion": "マンション", "mansion_ex30": "マンション(30㎡未満除く)"}
STARTS_LABELS = {"owner": "持家", "rental": "貸家", "sale_mansion": "分譲マンション", "sale_house": "分譲戸建", "company": "給与住宅"}
# 着工統計は対象都市のある都道府県と、全国・三大都市圏だけ載せる
STARTS_REGIONS = ["全国", "首都圏", "中部圏", "近畿圏", "北海道", "宮城", "埼玉", "千葉", "東京", "神奈川", "愛知", "京都", "大阪",
                  "兵庫", "広島", "福岡"]


def market_json() -> dict:
    """レインズの月次・価格帯別、既存住宅販売量指数、着工統計を、それぞれ共通の月(四半期)の並びに揃える。"""
    reins = pd.read_parquet(PROCESSED / "reins.parquet")
    bands = pd.read_parquet(PROCESSED / "reins_bands.parquet")
    sales = pd.read_parquet(PROCESSED / "sales_index.parquet")
    starts = pd.read_parquet(PROCESSED / "starts.parquet")

    def vals(s: pd.Series, index: list, digits: int = 2) -> list:
        return [None if pd.isna(v) else round(float(v), digits) for v in s.reindex(index)]

    months = list(pd.period_range(reins["month"].min(), reins["month"].max(), freq="M").strftime("%Y-%m"))
    regions = list(reins["region"].cat.categories)  # PDF の掲載順(首都圏 → 都県 → その内訳)
    notes = reins.groupby("region", observed=True)["region_note"].last().to_dict()
    series = {}
    for (kind, region, status), g in reins.groupby(["kind", "region", "status"], observed=True):
        g = g.set_index("month")
        cols = [c for c in ("n", "unit_price", "price", "age") if g[c].notna().any()]
        series.setdefault(kind, {}).setdefault(region, {})[status] = {c: vals(g[c], months) for c in cols}
    quarters = list(pd.period_range(bands["quarter"].min(), bands["quarter"].max(), freq="Q").strftime("%Y-Q%q"))
    band_labels, band_n = {}, {}
    for kind, g in bands[bands["band"] != "計"].groupby("kind"):
        labels = list(dict.fromkeys(g["band"]))
        if g.groupby(["region", "status", "quarter"], observed=True).size().nunique() != 1:
            raise SystemExit(f"レインズの価格帯の区分が時期によって違います: {kind}")
        band_labels[kind] = labels
        for (region, status), h in g.groupby(["region", "status"], observed=True):
            w = h.pivot(index="quarter", columns="band", values="n").reindex(index=quarters, columns=labels)
            band_n.setdefault(kind, {}).setdefault(region, {})[status] = [vals(w[b], quarters, 0) for b in labels]

    s_months = list(pd.period_range(sales["month"].min(), sales["month"].max(), freq="M").strftime("%Y-%m"))
    st_months = list(pd.period_range(starts["month"].min(), starts["month"].max(), freq="M").strftime("%Y-%m"))
    missing = sorted(set(STARTS_REGIONS) - set(starts["region"]))
    if missing:
        raise SystemExit(f"着工統計に地域がありません: {missing}")
    return {
        "kinds": MARKET_KINDS, "status": REINS_STATUS,
        "reins": {"month": months, "regions": regions, "region_notes": notes, "series": series,
                  "quarter": quarters, "bands": band_labels, "band_n": band_n},
        "sales_index": {"month": s_months, "labels": SALES_INDEX_LABELS, "regions": list(dict.fromkeys(sales["region"])),
                        "series": {r: {c: vals(g.set_index("month")[c], s_months) for c in SALES_INDEX_LABELS}
                                   | {f"{c}_n": vals(g.set_index("month")[f"{c}_n"], s_months, 0) for c in SALES_INDEX_LABELS}
                                   for r, g in sales.groupby("region")}},
        "starts": {"month": st_months, "labels": STARTS_LABELS, "regions": STARTS_REGIONS,
                   "series": {r: {c: vals(g.set_index("month")[c], st_months, 0) for c in ["total", *STARTS_LABELS]}
                              for r, g in starts[starts["region"].isin(STARTS_REGIONS)].groupby("region")}},
    }


# 「ローン・金利」タブ。商品の金利と金利タイプ別の利用割合は、公式ページ・PDFで確認した値を data/reference/ に置いている
REFERENCE_DIR = ROOT / "data" / "reference"
CHINTAI_LABELS = {"limited_35": "35年固定(繰上返済制限あり)", "limited_15": "15年固定(繰上返済制限あり)",
                  "free_35": "35年固定(繰上返済制限なし)", "free_15": "15年固定(繰上返済制限なし)"}
BOJ_LOAN_LABELS = {"housing_new": "住宅ローン(住宅資金)", "rental_new": "アパートローン等(個人による貸家業の設備資金)"}


def loan_json() -> dict:
    chintai = pd.read_parquet(PROCESSED / "chintai_rates.parquet")
    boj = pd.read_parquet(PROCESSED / "boj_loans.parquet")
    products = pd.read_csv(REFERENCE_DIR / "loan_products.csv", dtype={"as_of": str})
    share = pd.read_csv(REFERENCE_DIR / "jhf_rate_type_share.csv")
    clean = lambda v: None if pd.isna(v) else v  # noqa: E731
    return {
        "chintai": {"month": chintai["month"].tolist(), "labels": CHINTAI_LABELS,
                    "rates": {c: [clean(v) for v in chintai[c]] for c in CHINTAI_LABELS}},
        "boj_loans": {"quarter": boj["quarter"].tolist(), "labels": BOJ_LOAN_LABELS,
                      "values": {c: [clean(v) for v in boj[c]] for c in BOJ_LOAN_LABELS}},
        "products": [{k: clean(v) for k, v in r.items()} for r in products.to_dict("records")],
        "rate_type_share": {c: [clean(v) for v in share[c]] for c in share.columns},
    }


# 「人口」タブ。主要都市 = 東京23区 + 政令指定都市(市の合計の表がある)
MAJOR_CITIES = {"13100": "東京23区", "01100": "札幌市", "04100": "仙台市", "11100": "さいたま市", "12100": "千葉市", "14100": "横浜市",
                "14130": "川崎市", "14150": "相模原市", "15100": "新潟市", "22100": "静岡市", "22130": "浜松市", "23100": "名古屋市",
                "26100": "京都市", "27100": "大阪市", "27140": "堺市", "28100": "神戸市", "33100": "岡山市", "34100": "広島市",
                "40100": "北九州市", "40130": "福岡市", "43100": "熊本市"}
NATION = "00000"
# 推計と境界で単位が違う地域は、境界をまとめて推計の単位で塗る
# - 福島県の浜通り13市町村は「浜通り地域」(07999)としてまとめて推計されている(社人研の注記)
# - 浜松市は2024年に7区→3区に再編。推計は旧区なので、新しい区の境界は市全体(22130)にまとめる
MAP_MERGE = {**{c: "07999" for c in ["07204", "07209", "07212", "07541", "07542", "07543", "07544", "07545", "07546", "07547", "07548",
                                       "07561", "07564"]},
             **{c: "22130" for c in ["22138", "22139", "22140"]}}


def population_json() -> tuple[dict, dict]:
    """男女・5歳階級別の人口(地域ごとに 年×年齢 の行列を平らにしたもの)と、地図用の境界(人口のある市区町村だけ)。

    全国は都道府県の合計、東京23区は区の合計。その年の値がそろわない地域(2010年は12都道府県だけ取得)は null。
    """
    d = pd.read_parquet(PROCESSED / "population_detail.parquet")
    areas = pd.read_parquet(PROCESSED / "population_areas.parquet").set_index("code")
    years, ages = sorted(d["year"].unique()), sorted(d["age"].unique())
    prefs = [c for c in areas.index if c.endswith("000")]
    wards23 = [f"131{i:02d}" for i in range(1, 24)]

    def total(codes: list[str]) -> pd.DataFrame:
        g = d[d["code"].isin(codes)]
        n = g.groupby(["sex", "year"])["code"].nunique()
        ok = n[n == len(codes)].index  # すべての地域がそろう年だけ合計する
        return g.set_index(["sex", "year"]).loc[ok].reset_index().groupby(["sex", "year", "age"], as_index=False)["population"].sum()

    frames = {c: g for c, g in d.groupby("code")}
    frames[NATION] = total(prefs)
    frames["13100"] = total(wards23)
    pop = {}
    for code, g in frames.items():
        w = g.pivot_table(index=["sex", "year"], columns="age", values="population").reindex(columns=ages)
        pop[code] = {s: [None if pd.isna(v) else int(round(v)) for y in years
                         for v in (w.loc[(s, y)] if (s, y) in w.index else [np.nan] * len(ages))] for s in ("m", "f")}
    names = {**areas["name"].to_dict(), NATION: "全国", **MAJOR_CITIES}
    missing = [c for c in MAJOR_CITIES if c not in pop]
    if missing:
        raise SystemExit(f"人口の推計に主要都市がありません: {missing}")
    geo = {}
    for code, rings in json.loads((PROCESSED / "boundaries.json").read_text(encoding="utf-8")).items():
        geo.setdefault(MAP_MERGE.get(code, code), []).extend(rings)
    no_pop = sorted(set(geo) - set(pop))
    print(f"境界に人口がない地域 {len(no_pop)}: {no_pop[:10]}")
    geo = {c: v for c, v in geo.items() if c in pop}
    return {
        "years": [int(y) for y in years], "ages": [int(a) for a in ages], "actual_last": int(d.loc[d["actual"], "year"].max()),
        "names": {c: names.get(c, c) for c in pop}, "major": list(MAJOR_CITIES), "prefs": prefs, "nation": NATION,
        "map_areas": list(geo), "pop": pop,
    }, geo


def vacancy_json(map_areas: list[str], names: dict[str, str]) -> dict:
    """賃貸の空室率(住宅・土地統計調査)。地図の市区町村と、時系列のある全国・都道府県・大都市について 年ごとの値(なければ null)。"""
    d = pd.read_parquet(PROCESSED / "vacancy.parquet")
    years = sorted(int(y) for y in d["year"].unique())
    series = d[d["code"].isin(set(map_areas) | set(MAJOR_CITIES) | {NATION}) | d["code"].str.endswith("000")]
    w = {k: series.pivot(index="code", columns="year", values=k).reindex(columns=years) for k in ("rate", "rental_vacant", "rented")}
    as_list = lambda row, nd: [None if pd.isna(v) else round(float(v), nd) for v in row]
    missing = [c for c in MAJOR_CITIES if c not in w["rate"].index]
    if missing:
        raise SystemExit(f"空室率に主要都市がありません: {missing}")
    return {
        "years": years, "map_years": list(VACANCY_YEARS), "sale_until": VACANCY_SALE_UNTIL, "nation": NATION, "major": list(MAJOR_CITIES),
        "prefs": [c for c in w["rate"].index if c.endswith("000") and c != NATION],
        "names": {c: names.get(c, c) for c in [*w["rate"].index, *map_areas]}, "map_areas": map_areas,
        "rate": {c: as_list(r, 4) for c, r in w["rate"].iterrows()},
        "rental_vacant": {c: as_list(r, 0) for c, r in w["rental_vacant"].iterrows()},
        "rented": {c: as_list(r, 0) for c, r in w["rented"].iterrows()},
    }


def long_json() -> dict:
    """長期推移タブ: 都市別の地価指数(地価公示、最新年=100)・家賃指数(消費者物価、2020年=100)と、全国の物価・市街地価格指数を共通の年の並びに。

    家賃の水準(円/月)は住宅・土地統計調査の区×面積区分の平均家賃を戸数で加重平均した都市の平均で、指数を掛けて過去の目安にする。
    金利は1882年からの月次(系列ごとに始まりが違い、欠けた月は null)。
    """
    city = pd.read_parquet(PROCESSED / "long_city.parquet")
    nation = pd.read_parquet(PROCESSED / "long_nation.parquet").set_index("year")
    rent = pd.read_parquet(PROCESSED / "rent.parquet")
    rates = pd.read_parquet(PROCESSED / "long_rates.parquet").set_index("month")
    years = list(range(int(min(city["year"].min(), nation.index.min())), int(max(city["year"].max(), nation.index.max())) + 1))
    vals = lambda s, d=4: [None if pd.isna(v) else round(float(v), d) for v in s.reindex(years)]  # noqa: E731
    by = {c: g.set_index("year") for c, g in city.groupby("city")}
    cities = [c for c in CITIES_NORTH_TO_SOUTH if c in by]
    rent_level = {c: float(np.average(g["rent"], weights=g["n_units"])) for c, g in rent.groupby("city")}
    land_last = int(city.dropna(subset=["land"])["year"].max())
    return {
        "years": years, "cities": cities, "land_last_year": land_last, "rent_survey_year": 2023,
        "land": {c: vals(by[c]["land"], 3) for c in cities},
        "land_n": {c: [int(v) for v in by[c]["land_n"].reindex(years).fillna(0)] for c in cities},
        "land_level": {c: round(float(by[c].at[land_last, "land_level"])) for c in cities},
        "rent": {c: vals(by[c]["rent"], 2) for c in cities},
        "rent_level": {c: round(rent_level[c]) for c in cities if c in rent_level},
        "nation": {k: vals(nation[k], 2) for k in ("cpi", "jrei_six", "jrei_tokyo", "jrei_nation")},
        "rates": {"month": list(rates.index), "labels": LONG_RATE_LABELS,
                  "values": {k: [None if pd.isna(v) else round(float(v), 3) for v in rates[k]] for k in LONG_RATE_LABELS}},
    }


LONG_RATE_LABELS = {
    "policy_rate": "公定歩合(基準割引率および基準貸付利率)", "call_collateral": "有担保コールレート翌日物(月平均)",
    "call_rate": "無担保コールレート O/N(月平均)", "prime_long": "長期プライムレート", "prime_short": "短期プライムレート(最頻値)",
    "jgb10": "10年国債利回り(月平均)", "lend_stock_long": "貸出約定平均金利 ストック/長期",
}
# 長期推移タブの都市の並び(北から)
CITIES_NORTH_TO_SOUTH = ["札幌市", "仙台市", "さいたま市", "千葉市", "東京23区", "川崎市", "横浜市", "相模原市", "名古屋市", "京都市", "大阪市",
                         "堺市", "神戸市", "広島市", "北九州市", "福岡市"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", help="公開URL(末尾/)。指定すると docs/index.html の OGP の絶対URLを書き換える")
    args = ap.parse_args()

    out = SITE / "data"
    out.mkdir(parents=True, exist_ok=True)
    tx = pd.read_parquet(PROCESSED / "transactions.parquet")
    geo = pd.read_parquet(PROCESSED / "district_geo.parquet")
    cities = list(tx[~tx["kind"].isin(["bldg_rc", "bldg_wood"])]["city"].value_counts().index)  # 都市の並び(ファイル番号)はマンション・戸建の件数順
    # カテゴリ変数の水準は種別ごとに全都市共通の番号にする(全都市プールで連結できるように)
    levels = {k: {c: list(tx[tx["kind"] == k][c].value_counts().index) for c in VARIABLES[k]} for k in KINDS}

    datasets = {k: {} for k in KINDS}
    for (kind, city), g in tx.groupby(["kind", "city"]):
        stem = f"{kind}_{cities.index(city):02d}"
        info, blob = dataset(g, kind, levels[kind], geo)
        (out / f"{stem}.json").write_text(json.dumps(info, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        (out / f"{stem}.bin.gz").write_bytes(blob)
        datasets[kind][city] = {"file": stem, "n": len(g), "wards": info["wards"]}
        print(f"{stem} {city} n={len(g):,} {len(blob) / 1e6:.2f}MB")

    pop = pd.read_parquet(PROCESSED / "population.parquet")
    rent = pd.read_parquet(PROCESSED / "rent.parquet")
    rent_age = pd.read_parquet(PROCESSED / "rent_age.parquet")
    meta = {
        "cities": cities, "all_label": ALL, "kinds": KINDS, "datasets": datasets,
        "variables": VARIABLES, "labels": CATEGORICALS, "levels": levels, "reference": REFERENCE,
        "district_lambda": DISTRICT_LAMBDA, "no_district": NO_DISTRICT,
        "year_min": int(tx["year"].min()), "year_max": int(tx["year"].max()), "n_total": len(tx),
        "population_last_actual_year": 2020,
        "population": {w: {"year": g["year"].tolist(), "population": g["population"].round().astype(int).tolist()}
                       for w, g in pop.groupby("ward")},
        "rent": rent[["ward", "area_min", "area_max", "rent", "n_units"]].to_dict("records"),
        "rent_age": {c: {"age": g["age"].tolist(), "factor": g["factor"].round(4).tolist()}
                     for c, g in rent_age.groupby("city")},
        "population_elasticity": {k: population_elasticity(tx, k) for k in KINDS},
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print("meta.json", round((out / "meta.json").stat().st_size / 1e6, 2), "MB")
    (out / "rates.json").write_text(json.dumps(rates_json(), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print("rates.json", round((out / "rates.json").stat().st_size / 1e6, 2), "MB")
    (out / "market.json").write_text(json.dumps(market_json(), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print("market.json", round((out / "market.json").stat().st_size / 1e6, 2), "MB")
    pop, geo = population_json()
    (out / "population.json").write_text(json.dumps(pop, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    (out / "boundaries.json").write_text(json.dumps(geo, separators=(",", ":")), encoding="utf-8")
    print("population.json", round((out / "population.json").stat().st_size / 1e6, 2), "MB, boundaries.json",
          round((out / "boundaries.json").stat().st_size / 1e6, 2), "MB")
    (out / "vacancy.json").write_text(json.dumps(vacancy_json(pop["map_areas"], pop["names"]), ensure_ascii=False, separators=(",", ":")),
                                      encoding="utf-8")
    print("vacancy.json", round((out / "vacancy.json").stat().st_size / 1e6, 2), "MB")
    (out / "loan.json").write_text(json.dumps(loan_json(), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print("loan.json", round((out / "loan.json").stat().st_size / 1e6, 2), "MB")
    (out / "long.json").write_text(json.dumps(long_json(), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print("long.json", round((out / "long.json").stat().st_size / 1e6, 2), "MB")

    if args.base_url:
        index = SITE / "index.html"
        html = re.sub(r'(<meta property="og:(?:url|image)" content=")[^"]*?((?:ogp\.png)?")',
                      lambda m: m.group(1) + args.base_url + m.group(2), index.read_text(encoding="utf-8"))
        index.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
