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

from .model import CATEGORICALS, Spec, fit

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
SITE = ROOT / "docs"

ALL = "全都市"
KINDS = {"mansion": "中古マンション", "house": "中古戸建(土地と建物)"}
# 種別ごとの説明変数(すべて投入する)。JS もこの定義を meta.json 経由で使う
VARIABLES = {
    "mansion": ["structure", "renovated", "layout", "zoning", "far", "future_use", "source", "quarter"],
    "house": ["structure", "zoning", "far", "future_use", "source", "quarter", "land_shape", "road_dir", "road_type",
              "road_width", "frontage", "region"],
}
# 切片に対応する「基準の物件」。連続変数はこの値からの差で入れ、カテゴリ項目は最頻の水準を基準にする
# (マンションは 20㎡ に合わせて間取りの基準を 1K に固定)
REFERENCE = {
    "mansion": {"ref_age": 10, "ref_area": 20, "ref_station": 10, "base_levels": {"layout": "1K"}},
    "house": {"ref_age": 10, "ref_area": 100, "ref_land": 100, "ref_station": 10, "base_levels": {}},
}
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
    if kind == "house":
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
    spec = Spec(use_land=kind == "house", use_population=True, categoricals=tuple(VARIABLES[kind]),
                fixed_effects="ward+cityyear", ref_age=ref["ref_age"], ref_area=ref["ref_area"],
                ref_land=ref.get("ref_land", 1.0), ref_station=ref["ref_station"],
                base_levels=tuple(ref["base_levels"].items()))
    f = fit(tx[tx["kind"] == kind], spec, cluster=True)
    b, se = f.coef("ln_pop")
    return {"coef": b, "se": se, "ci_low": b - 1.96 * se, "ci_high": b + 1.96 * se, "n": f.n, "r2": f.r2,
            "se_type": f.se_type}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", help="公開URL(末尾/)。指定すると docs/index.html の OGP の絶対URLを書き換える")
    args = ap.parse_args()

    out = SITE / "data"
    out.mkdir(parents=True, exist_ok=True)
    tx = pd.read_parquet(PROCESSED / "transactions.parquet")
    geo = pd.read_parquet(PROCESSED / "district_geo.parquet")
    cities = list(tx["city"].value_counts().index)
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

    if args.base_url:
        index = SITE / "index.html"
        html = re.sub(r'(<meta property="og:(?:url|image)" content=")[^"]*?((?:ogp\.png)?")',
                      lambda m: m.group(1) + args.base_url + m.group(2), index.read_text(encoding="utf-8"))
        index.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
