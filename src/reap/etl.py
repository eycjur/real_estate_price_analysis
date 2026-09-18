"""data/raw の公的データを分析用の parquet に整形する。

- 取引: 国土交通省 不動産情報ライブラリ「不動産取引価格情報」CSV (zip)
- 人口: 国立社会保障・人口問題研究所「日本の地域別将来推計人口」
        (平成25年・平成30年・令和5年推計。各推計の基準年は国勢調査の実績値)
- 家賃: 総務省統計局「令和5年住宅・土地統計調査」第130表・第107-2表
- 位置: 国土交通省「位置参照情報」大字・町丁目レベル(地区の代表点。地図表示用)

使い方: uv run python -m reap.etl
"""

from __future__ import annotations

import io
import re
import unicodedata
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

# 分析対象: 東京23区 + 政令指定都市(取得した12都道府県内)
CITIES = [
    "札幌市", "仙台市", "さいたま市", "千葉市", "東京23区", "横浜市", "川崎市", "相模原市",
    "名古屋市", "京都市", "大阪市", "堺市", "神戸市", "広島市", "北九州市", "福岡市",
]

KIND_LABEL = {"中古マンション等": "mansion", "宅地(土地と建物)": "house"}

# 「30分～60分」のような幅のある表記は区間の中央値に置き換える
STATION_RANGE = {"30分~60分": 45, "1H~1H30": 75, "1H30~2H": 105, "2H~": 120}


def city_of(code: int, name: str) -> str | None:
    if 13101 <= code <= 13123:
        return "東京23区"
    for c in CITIES:
        if name.startswith(c):
            return c
    return None


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _top(s: pd.Series, n: int) -> pd.Series:
    """頻度上位 n 水準を残し、それ以外は「その他」、欠損は「不明」にまとめる。"""
    keep = set(s.value_counts().index[:n])
    return s.where(s.isin(keep) | s.isna(), "その他").fillna("不明")


def _bins(s: pd.Series, edges: list[float], unit: str) -> pd.Series:
    """数値を区間カテゴリにする(欠損は「不明」)。成約価格情報に無い項目でも標本を落とさないため。"""
    labels = [f"{a:g}〜{b:g}{unit}" for a, b in zip(edges[:-2], edges[1:-1])] + [f"{edges[-2]:g}{unit}以上"]
    return pd.cut(_num(s), edges, right=False, labels=labels).astype(object).fillna("不明")


ROAD_PUBLIC = {"国道", "都道", "道道", "府道", "県道", "市道", "区道", "町道", "村道", "公道", "道路", "区画街路"}


def load_transactions() -> pd.DataFrame:
    frames = []
    for z in sorted(RAW.glob("*.zip")):
        with zipfile.ZipFile(z) as zf:
            for n in zf.namelist():
                if n.lower().endswith(".csv"):
                    frames.append(pd.read_csv(io.BytesIO(zf.read(n)), encoding="cp932", dtype=str))
    # 全角の記号・英数字(「：」「㎡」「ＲＣ」等)を NFKC で半角に揃える
    for f in frames:
        f.columns = [unicodedata.normalize("NFKC", c) for c in f.columns]
    raw = pd.concat(frames, ignore_index=True)
    raw = raw.apply(lambda s: s.map(lambda v: unicodedata.normalize("NFKC", v) if isinstance(v, str) else v))

    df = pd.DataFrame({
        "kind": raw["種類"].map(KIND_LABEL),
        "ward_code": _num(raw["市区町村コード"]),
        "pref": raw["都道府県名"],
        "ward": raw["市区町村名"],
        "district": raw["地区名"],
        "station": raw["最寄駅:名称"],
        "station_min": _num(raw["最寄駅:距離(分)"].replace(STATION_RANGE)),
        "price": _num(raw["取引価格(総額)"]),
        "area": _num(raw["面積(m2)"]),
        "built_year": _num(raw["建築年"].str.extract(r"(\d{4})年")[0]),
        "structure": raw["建物の構造"],
        # 成約価格情報には改装の記載がないため「不明」を独立の区分にする
        "renovated": raw["改装"].fillna("不明"),
        "remarks": raw["取引の事情等"],
        "use": raw["用途"],
        "source": raw["価格情報区分"].replace({"不動産取引価格情報": "取引価格(アンケート)", "成約価格情報": "成約価格(レインズ)"}),
        "layout": _top(raw["間取り"], 12),
        "zoning": _top(raw["都市計画"], 12),
        "far": _bins(raw["容積率(%)"], [0, 150, 250, 350, 450, 650, 9999], "%"),
        "future_use": _top(raw["今後の利用目的"], 2),
    })
    if "土地の形状" in raw:  # 戸建(土地と建物)のみにある項目
        road = raw["前面道路:種類"]
        df["land_shape"] = _top(raw["土地の形状"], 9)
        df["road_dir"] = _top(raw["前面道路:方位"], 9)
        df["road_type"] = road.where(road.isna(), np.where(road == "私道", "私道", np.where(road.isin(ROAD_PUBLIC), "公道", "その他")))
        df["road_type"] = df["road_type"].fillna("不明")
        df["road_width"] = _bins(raw["前面道路:幅員(m)"], [0, 4, 6, 8, 12, 999], "m")
        df["frontage"] = _bins(raw["間口"], [0, 5, 8, 12, 999], "m")
        df["region"] = raw["地域"].fillna("不明")
    if "延床面積(m2)" in raw:
        df["floor_area"] = _num(raw["延床面積(m2)"])
    q = raw["取引時期"].str.extract(r"(\d{4})年第(\d)四半期")
    df["year"] = _num(q[0])
    df["quarter"] = "Q" + q[1]
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    df = df[df["kind"].notna() & df["ward_code"].notna()].copy()
    df["ward_code"] = df["ward_code"].astype(int)
    df["city"] = [city_of(c, w) for c, w in zip(df["ward_code"], df["ward"])]
    df = df[df["city"].notna()]
    # 戸建は建物(延床)面積を、マンションは専有面積を規模の指標にする
    if "floor_area" in df:
        df["land_area"] = np.where(df["kind"] == "house", df["area"], np.nan)
        df["area"] = np.where(df["kind"] == "house", df["floor_area"], df["area"])
        df = df.drop(columns="floor_area")
    df = df[(df["kind"] != "house") | df["land_area"].between(20, 1000)]
    df["age"] = df["year"] - df["built_year"]
    # 回帰に必須の項目が欠けるもの、私道・調停など特殊事情の取引、極端な値を除く
    need = ["price", "area", "built_year", "year", "station_min"]
    df = df.dropna(subset=need)
    df = df[df["remarks"].isna()]
    # 店舗・事務所・共同住宅(一棟)などを除き、住宅(用途未記載を含む)に限る
    df = df[df["use"].isna() | (df["use"] == "住宅")]
    df = df[(df["age"] >= 0) & (df["age"] <= 60) & (df["area"] >= 10) & (df["area"] <= 300) & (df["price"] > 0)]
    # 主要構造以外(少数例や未記載)は「その他」にまとめる
    main = {"mansion": ["RC", "SRC"], "house": ["木造", "軽量鉄骨造", "鉄骨造", "RC"]}
    df["structure"] = [s if s in main[k] else "その他" for s, k in zip(df["structure"], df["kind"])]
    df["district_key"] = df["ward"] + " " + df["district"].fillna("(地区不明)")
    df["unit_price"] = df["price"] / df["area"]
    # 種別×都市×年ごとに㎡単価の上下0.5%を外れ値として除く
    lo = df.groupby(["kind", "city", "year"])["unit_price"].transform(lambda s: s.quantile(0.005))
    hi = df.groupby(["kind", "city", "year"])["unit_price"].transform(lambda s: s.quantile(0.995))
    df = df[(df["unit_price"] >= lo) & (df["unit_price"] <= hi)]
    print(f"transactions: raw={n0:,} -> cleaned={len(df):,}")
    return df.drop(columns=["remarks", "use"]).reset_index(drop=True)


def _read_sheets(path: Path):
    if path.suffix == ".xlsx":
        for name, sh in pd.read_excel(path, sheet_name=None, header=None, engine="openpyxl").items():
            yield name, sh
    else:
        for name, sh in pd.read_excel(path, sheet_name=None, header=None, engine="xlrd").items():
            yield name, sh


def load_population() -> pd.DataFrame:
    """区市町村別の総人口(実績+推計)を long 形式で返す。新しい推計を優先する。"""
    rows = []
    for vintage in (2013, 2018, 2023):
        for f in sorted((RAW / f"ipss{vintage}").glob("*.xls*")):
            for name, sh in _read_sheets(f):
                m = re.match(r"\s*(\d{4,5})", str(name))  # 北海道は先頭0が落ちた4桁
                if not m:
                    continue
                hdr = sh.index[sh[0] == "男女計"]
                if len(hdr) == 0:
                    continue
                h = hdr[0]
                base = vintage - 3  # 基準年(国勢調査実績): 2010 / 2015 / 2020
                for col in range(1, sh.shape[1]):
                    ym = re.match(r"(\d{4})年", str(sh.iat[h, col]))
                    if not ym:
                        break
                    rows.append((int(m.group(1)), int(ym.group(1)), float(sh.iat[h + 1, col]),
                                 vintage, int(ym.group(1)) == base))
    pop = pd.DataFrame(rows, columns=["ward_code", "year", "population", "vintage", "actual"])
    # 同じ年は 実績 > 新しい推計 の順に採用
    pop = (pop.sort_values(["actual", "vintage"]).drop_duplicates(["ward_code", "year"], keep="last")
              .sort_values(["ward_code", "year"]).reset_index(drop=True))
    return backcast_wards(pop)


def backcast_wards(pop: pd.DataFrame) -> pd.DataFrame:
    """過去の推計に区別の表がない市(さいたま市・相模原市)は、最古の区人口×市計の伸びで遡る。

    区のシェアを固定する近似。区FE+都市×年FEの人口モデルでは市計の動きは吸収されるため、
    この補完分は人口弾力性の推定には寄与しない(欠損で取引を落とさないための措置)。
    """
    first = pop["year"].min()
    by_code = {c: g.set_index("year")["population"] for c, g in pop.groupby("ward_code")}
    extra = []
    for code, s in by_code.items():
        if s.index.min() == first:
            continue
        # 親の市コード: 14152→14150、末尾0の区(11110 岩槻区)は 11100
        parents = [c for c in (code // 10 * 10, code // 100 * 100) if c != code and c in by_code]
        if not parents:
            continue
        parent = by_code[parents[0]]
        y0 = s.index.min()
        for y in parent.index[parent.index < y0]:
            extra.append((code, y, s[y0] * parent[y] / parent[y0], 0, False))
    add = pd.DataFrame(extra, columns=pop.columns)
    return pd.concat([pop, add]).sort_values(["ward_code", "year"]).reset_index(drop=True)


def annual_population(pop: pd.DataFrame) -> pd.DataFrame:
    """5年刻みの人口を年次に線形補間する。"""
    out = []
    for code, g in pop.groupby("ward_code"):
        years = np.arange(g["year"].min(), g["year"].max() + 1)
        out.append(pd.DataFrame({
            "ward_code": code, "year": years,
            "population": np.interp(years, g["year"], g["population"]),
        }))
    return pd.concat(out, ignore_index=True)


RENT_SURVEY_YEAR = 2023
AREA_BRACKETS = {"1": (0, 29), "2": (30, 49), "3": (50, 69), "4": (70, 99), "5": (100, 149), "6": (150, 9999)}


def load_rent(wards: pd.DataFrame) -> pd.DataFrame:
    """第130表: 区×延べ面積区分別の民営借家(非木造共同住宅)の平均家賃[円/月](家賃0円を含まない)。"""
    d = pd.read_excel(RAW / "rent" / "t130.xlsx", header=None, skiprows=9)
    d = d[d[2].astype(str).str.startswith("4_")]  # 4_共同住宅(非木造)
    out = pd.DataFrame({
        "ward_code": _num(d[1].astype(str).str.extract(r"^(\d{5})_")[0]),
        "bracket": d[3].astype(str).str[0],
        "n_units": _num(d[4]),
        "rent": _num(d[d.columns[-1]]),  # 「-」(該当なし)は NaN
    })
    out = out[out["bracket"].isin(list(AREA_BRACKETS))].dropna(subset=["ward_code", "rent"])
    out["ward_code"] = out["ward_code"].astype(int)
    out["area_min"] = out["bracket"].map(lambda b: AREA_BRACKETS[b][0])
    out["area_max"] = out["bracket"].map(lambda b: AREA_BRACKETS[b][1])
    return out.merge(wards, on="ward_code")


def load_rent_age() -> pd.DataFrame:
    """第107-2表: 都市別・建築時期別の民営借家(非木造)1畳当たり家賃を、都市平均に対する比にする。"""
    d = pd.read_excel(RAW / "rent" / "t107_2.xlsx", header=None, skiprows=9)
    name = d[1].astype(str).str.split("_").str[1].replace({"特別区部": "東京23区"})
    d = d.assign(city=name, rent=_num(d[9]))  # 列9: 132_民営借家(非木造)
    d = d[d["city"].isin(CITIES)]
    total = d[d[2].astype(str).str.startswith("00_")].set_index("city")["rent"]
    rows = []
    for _, r in d.iterrows():
        m = re.match(r"\d+_(\d{4})～(\d{4})年", str(r[2]))
        if not m or pd.isna(r["rent"]):
            continue
        mid = (int(m.group(1)) + min(int(m.group(2)), RENT_SURVEY_YEAR)) / 2
        rows.append((r["city"], RENT_SURVEY_YEAR - mid, r["rent"] / total[r["city"]]))
    return pd.DataFrame(rows, columns=["city", "age", "factor"]).sort_values(["city", "age"])


def _kanji_int(k: str) -> int:
    """「二十四」のような99以下の漢数字を整数にする。"""
    d = {c: i for i, c in enumerate("一二三四五六七八九", 1)}
    tens, _, ones = k.rpartition("十") if "十" in k else ("", "", k)
    return (d.get(tens, 1) * 10 if "十" in k else 0) + d.get(ones, 0)


def load_district_geo(districts: pd.DataFrame) -> pd.DataFrame:
    """取引データの地区(町名)ごとの代表点。位置参照情報は丁目単位なので、丁目を外した町名で平均する。"""
    frames = []
    for z in sorted((RAW / "isj").glob("*.zip")):
        with zipfile.ZipFile(z) as zf:
            name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
            frames.append(pd.read_csv(io.BytesIO(zf.read(name)), encoding="cp932", dtype=str))
    g = pd.concat(frames, ignore_index=True)
    # 丁目(堺市は「丁」)を外して町名に揃える。札幌の「北一条西」は取引データでは「北1条西」と書かれる
    town = g["大字町丁目名"].str.replace(r"[一二三四五六七八九十]+丁目?$", "", regex=True)
    arabic = town.str.replace(r"([一二三四五六七八九十]+)条", lambda m: f"{_kanji_int(m.group(1))}条", regex=True)
    g = g.assign(ward=g["市区町村名"], lat=_num(g["緯度"]), lon=_num(g["経度"]), district=town)
    # 堺市「一条通」などは取引データでも漢数字のままなので、両方の表記を候補にする
    g = pd.concat([g, g.assign(district=arabic)[arabic != town]], ignore_index=True)
    geo = g.groupby(["ward", "district"], as_index=False)[["lat", "lon"]].mean()
    out = districts.merge(geo, on=["ward", "district"], how="inner")
    print(f"district geo: matched {len(out):,} / {len(districts):,} districts")
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tx = clean(load_transactions())
    pop = annual_population(load_population())
    tx = tx.merge(pop, on=["ward_code", "year"], how="left")
    missing = tx["population"].isna().mean()
    print(f"population join: missing={missing:.2%}")
    tx.to_parquet(OUT / "transactions.parquet", index=False)
    wards = tx[["ward_code", "city", "ward"]].drop_duplicates("ward_code")
    pop.merge(wards, on="ward_code").to_parquet(OUT / "population.parquet", index=False)
    load_rent(wards).to_parquet(OUT / "rent.parquet", index=False)
    load_rent_age().to_parquet(OUT / "rent_age.parquet", index=False)
    districts = tx[["ward", "district", "district_key"]].dropna().drop_duplicates("district_key")
    load_district_geo(districts).to_parquet(OUT / "district_geo.parquet", index=False)
    print(tx.groupby(["kind", "city"]).size().unstack(0))


if __name__ == "__main__":
    main()
