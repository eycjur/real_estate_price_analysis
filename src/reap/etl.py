"""data/raw の公的データを分析用の parquet に整形する。

- 取引: 国土交通省 不動産情報ライブラリ「不動産取引価格情報」CSV (zip)
- 人口: 国立社会保障・人口問題研究所「日本の地域別将来推計人口」
        (平成25年・平成30年・令和5年推計。各推計の基準年は国勢調査の実績値)
- 家賃: 総務省統計局「令和5年住宅・土地統計調査」第130表・第107-2表
- 位置: 国土交通省「位置参照情報」大字・町丁目レベル(地区の代表点。地図表示用)
- 金利: 日本銀行(基準貸付利率・無担保コールレート・貸出約定平均金利・プライムレート)、財務省「国債金利情報」
- 不動産価格指数(住宅): 国土交通省(月次、季節調整済み)
- 市況: 東日本不動産流通機構「月例速報 Market Watch」(首都圏の成約・新規登録・在庫)、国土交通省「既存住宅販売量指数」
        「建築着工統計」(都道府県別・利用関係別の新設住宅着工戸数)
- 人口(人口タブ): 社人研の推計を全都道府県・男女5歳階級別に。地図の境界は国土数値情報「行政区域」(N03)
- ローン: 住宅金融支援機構「賃貸住宅融資 参考金利の推移」、日本銀行「貸出先別貸出金」(住宅資金・個人による貸家業の新規貸出)

使い方: uv run python -m reap.etl
"""

from __future__ import annotations

import csv
import html
import io
import json
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
# 一棟もの(土地と建物で用途が共同住宅だけの取引)は構造で分ける。築年数による価値の落ち方が構造で大きく違うため
BLDG_KIND = {"RC": "bldg_rc", "SRC": "bldg_rc", "木造": "bldg_wood", "軽量鉄骨造": "bldg_wood", "鉄骨造": "bldg_wood"}
LAND_KINDS = ("house", "bldg_rc", "bldg_wood")  # 土地と建物の取引(建物は延床面積、土地面積も使う)
# 種別ごとの面積(マンションは専有、それ以外は延床)・土地面積の範囲。一棟ものは規模が大きい
AREA_RANGE = {"mansion": (10, 300), "house": (10, 300), "bldg_rc": (30, 3000), "bldg_wood": (30, 3000)}
LAND_RANGE = {"house": (20, 1000), "bldg_rc": (30, 3000), "bldg_wood": (30, 3000)}

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
        # 容積率の消化率 = 延床 ÷ (土地面積 × 容積率)。100%超は今の規制では同じ規模に建て替えられない(既存不適格)可能性がある
        usage = df["floor_area"] / (df["area"] * _num(raw["容積率(%)"]) / 100) * 100
        df["far_usage"] = _bins(usage, [0, 50, 80, 100, 120, 1e9], "%")
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
    # 土地と建物の取引のうち、用途が共同住宅だけのものは一棟もの(マンション一棟・アパート一棟)。
    # マンション・戸建は用途が住宅(未記載を含む)に限り、店舗・事務所などとの併用や用途の混在は除く
    bldg = (df["kind"] == "house") & (df["use"] == "共同住宅")
    df.loc[bldg, "kind"] = df.loc[bldg, "structure"].map(BLDG_KIND)  # 混構造・ブロック造などは除く
    df = df[df["kind"].notna()]
    df = df[df["kind"].isin(["bldg_rc", "bldg_wood"]) | df["use"].isna() | (df["use"] == "住宅")]  # マンション・戸建は住宅用途だけ
    land = df["kind"].isin(LAND_KINDS)
    # 戸建・一棟は建物(延床)面積を、マンションは専有面積を規模の指標にする
    if "floor_area" in df:
        df["land_area"] = np.where(land, df["area"], np.nan)
        df["area"] = np.where(land, df["floor_area"], df["area"])
        df = df.drop(columns="floor_area")
    lo, hi = df["kind"].map({k: v[0] for k, v in LAND_RANGE.items()}), df["kind"].map({k: v[1] for k, v in LAND_RANGE.items()})
    df = df[~df["kind"].isin(LAND_KINDS) | ((df["land_area"] >= lo) & (df["land_area"] <= hi))]
    df["age"] = df["year"] - df["built_year"]
    # 回帰に必須の項目が欠けるもの、私道・調停など特殊事情の取引、極端な値を除く
    need = ["price", "area", "built_year", "year", "station_min"]
    df = df.dropna(subset=need)
    df = df[df["remarks"].isna()]
    amin, amax = df["kind"].map({k: v[0] for k, v in AREA_RANGE.items()}), df["kind"].map({k: v[1] for k, v in AREA_RANGE.items()})
    df = df[(df["age"] >= 0) & (df["age"] <= 60) & (df["area"] >= amin) & (df["area"] <= amax) & (df["price"] > 0)]
    # 主要構造以外(少数例や未記載)は「その他」にまとめる
    main = {"mansion": ["RC", "SRC"], "house": ["木造", "軽量鉄骨造", "鉄骨造", "RC"], "bldg_rc": ["RC", "SRC"], "bldg_wood": ["木造", "軽量鉄骨造", "鉄骨造"]}
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


IPSS_SEX = {"男": "m", "女": "f"}
AGE_TOP = 90  # 最上位の区分は推計の版で「90歳以上」「90～94歳・95歳～」と違うので 90歳以上 に揃える


def _ipss_sheet(sh: pd.DataFrame) -> list[tuple]:
    """社人研の推計の1シート(1地域)を (性別, 年, 年齢の下限, 人口) にする。

    「男」「女」のセルの右に「2020年 2025年 …」が並び、その下に「総数」「0～4歳」…の行が続く。
    令和5年推計は男女計・男・女が横に、それ以前は縦に並ぶ。「（再掲）」以降は使わない。
    """
    v = sh.to_numpy(dtype=object)
    out = []
    for r, c in zip(*np.nonzero(np.isin(v, list(IPSS_SEX)))):
        years = []
        while c + 1 + len(years) < v.shape[1] and isinstance(y := v[r, c + 1 + len(years)], str) and re.fullmatch(r"\d{4}年", y.strip()):
            years.append(int(y.strip()[:4]))
        rr = r + 1
        while rr < v.shape[0] and isinstance(v[rr, c], str):
            label = unicodedata.normalize("NFKC", v[rr, c]).strip()
            if label.startswith("(再掲") or "割合" in label:
                break
            if label != "総数":
                lb = min(int(re.match(r"\d+", label).group(0)), AGE_TOP)
                out += [(IPSS_SEX[v[r, c]], y, lb, float(v[rr, c + 1 + j])) for j, y in enumerate(years)]
            rr += 1
    return out


HAMADORI = ["07204", "07209", "07212", "07541", "07542", "07543", "07544", "07545", "07546", "07547", "07548", "07561", "07564"]
CENSUS_YEAR = 2025


def load_census_age(path: Path | None = None) -> pd.DataFrame:
    """令和7年国勢調査 人口等基本集計 第2-7表(男女・5歳階級別人口、不詳補完値)。全国・都道府県・市区町村(政令市は区も)。

    「地域識別コード」9 の行は2000年時点の市町村に組み替えた値なので使わない。90歳以上はまとめ、年齢「不詳」(補完後は0)は除く。
    福島県の浜通り13市町村は、社人研の推計の単位に合わせて「浜通り地域」(07999)の合計も作る。
    """
    sh = pd.read_excel(path or RAW / "census2025" / "age5.xlsx", header=None)
    head = next(i for i in range(len(sh)) if sh.iat[i, 0] == "国籍総数か日本人")
    items = sh.iloc[head - 3]
    d = sh.iloc[head + 1:]
    d = d[(d[0] == "0_国籍総数") & d[1].isin(["1_男", "2_女"]) & (d[2].astype(str) != "9")]
    rows = []
    for c, item in items.items():
        m = re.match(r"(\d\d)_(\d+)～", str(item)) or re.match(r"(\d\d)_(\d+)歳以上", str(item))
        if m and int(m.group(1)) <= 21:  # 01_0～4歳 … 21_100歳以上
            rows.append(pd.DataFrame({"code": d[7].astype(str).str.zfill(5).to_numpy(), "sex": d[1].map({"1_男": "m", "2_女": "f"}).to_numpy(),
                                      "age": min(int(m.group(2)), AGE_TOP), "population": _num(d[c]).to_numpy()}))
    out = pd.concat(rows).groupby(["code", "sex", "age"], as_index=False)["population"].sum()
    hama = out[out["code"].isin(HAMADORI)]
    if hama["code"].nunique() == len(HAMADORI):
        out = pd.concat([out, hama.groupby(["sex", "age"], as_index=False)["population"].sum().assign(code="07999")])
    print(f"census {CENSUS_YEAR}: {out['code'].nunique()} areas, 全国 {out[out['code'] == '00000']['population'].sum():,.0f}人")
    return out.assign(year=CENSUS_YEAR)


def load_population_detail() -> tuple[pd.DataFrame, pd.DataFrame]:
    """全都道府県・市区町村(政令市は区も)の男女・5歳階級別人口(実績+推計)と、地域の名前。

    同じ地域・年は 国勢調査の実績 > 新しい推計 の順に採用する(各推計の基準年が実績: 2010/2015/2020、2025年は令和7年国勢調査)。
    """
    rows, names = [], {}
    for vintage in (2013, 2018, 2023):
        for f in sorted((RAW / f"ipss{vintage}").glob("*.xls*")):
            for sheet, sh in _read_sheets(f):
                m = re.match(r"\s*(\d{4,5})[\s_]*(.*)", str(sheet))
                if not m:
                    continue
                code = m.group(1).zfill(5)
                name = re.sub(r"^\S+[都道府県]\s+", "", m.group(2).strip()) if not code.endswith("000") else m.group(2).strip()
                names[code] = (name, int(code[:2]))  # 新しい推計の名前で上書き
                base = vintage - 3
                rows += [(code, vintage, s, y, a, p, y == base) for s, y, a, p in _ipss_sheet(sh)]
    d = pd.DataFrame(rows, columns=["code", "vintage", "sex", "year", "age", "population", "actual"])
    d = d.groupby(["code", "vintage", "sex", "year", "age", "actual"], as_index=False)["population"].sum()  # 90歳以上をまとめる
    # 令和7年国勢調査の実績で、推計の2025年を置き換える(新しい市区町村(浜松市の新しい区など)は2025年だけの地域になる)
    d = pd.concat([d, load_census_age().assign(vintage=CENSUS_YEAR, actual=True)], ignore_index=True)
    d = (d.sort_values(["actual", "vintage"]).drop_duplicates(["code", "sex", "year", "age"], keep="last")
          .sort_values(["code", "sex", "year", "age"]).reset_index(drop=True))
    areas = pd.DataFrame([(c, n, p) for c, (n, p) in names.items()], columns=["code", "name", "pref"]).sort_values("code")
    print(f"population detail: {d['code'].nunique()} areas, {d['year'].min()}〜{d['year'].max()}")
    return d, areas.reset_index(drop=True)


def _simplify(pts: np.ndarray, tol: float) -> np.ndarray:
    """Douglas–Peucker 法で折れ線を間引く(tol は度)。"""
    keep = np.zeros(len(pts), bool)
    keep[[0, -1]] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = pts[i], pts[j]
        seg = b - a
        norm = np.hypot(*seg)
        rel = pts[i + 1:j] - a
        dist = np.abs(seg[0] * rel[:, 1] - seg[1] * rel[:, 0]) / norm if norm else np.hypot(rel[:, 0], rel[:, 1])
        k = int(np.argmax(dist))
        if dist[k] > tol:
            keep[i + 1 + k] = True
            stack += [(i, i + 1 + k), (i + 1 + k, j)]
    return pts[keep]


BOUNDARY_TOL = 0.004  # 約400m。全国〜都道府県の縮尺で見る地図用


def load_boundaries() -> dict[str, list]:
    """国土数値情報「行政区域」(N03)の市区町村(政令市は区)の境界を、間引いて {コード: [外周リング, …]} にする。

    1つの市区町村は島ごとの多数のポリゴンからなるので、間引いた後に3点未満になる小さな島は落とす
    (すべて落ちる市区町村は、いちばん大きいポリゴンだけ残す)。穴(内側のリング)は使わない。
    """
    shapes: dict[str, list] = {}
    for z in sorted((RAW / "n03").glob("N03-*_GML.zip")):
        with zipfile.ZipFile(z) as zf:
            gj = json.loads(zf.read(next(n for n in zf.namelist() if n.endswith(".geojson"))))
        for feat in gj["features"]:
            p, code = feat["properties"], feat["properties"]["N03_007"]
            if not code or p["N03_004"] == "所属未定地":
                continue
            polys = feat["geometry"]["coordinates"]
            polys = [polys] if feat["geometry"]["type"] == "Polygon" else polys
            shapes.setdefault(code, []).extend(np.asarray(poly[0]) for poly in polys)
    out = {}
    for code, rings in shapes.items():
        simple = [np.round(_simplify(r, BOUNDARY_TOL), 3) for r in rings]
        keep = [s for s in simple if len(np.unique(s, axis=0)) >= 3]
        if not keep:
            keep = [max(simple, key=len)]
        out[code] = [s.tolist() for s in keep]
    print(f"boundaries: {len(out)} areas, {sum(len(r) for v in out.values() for r in v):,} points")
    return out


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


RATES_FROM = "2000-01"  # 金利・指数はこの月以降を使う(取引データは2010年〜)


def _boj_csv(path: Path) -> pd.DataFrame:
    """日銀「主要時系列統計データ表」の CSV。月(YYYY-MM)を index、系列名称を列名にする。"""
    lines = path.read_text(encoding="cp932").splitlines()
    names = next(csv.reader([next(ln for ln in lines if ln.startswith('"系列名称"'))]))[1:]
    data = [ln for ln in lines if re.match(r"^\d{4}/\d{2},", ln)]
    d = pd.read_csv(io.StringIO("\n".join(data)), header=None, names=["month", *names], na_values=["", "ND", "NA"])
    d["month"] = d["month"].str.replace("/", "-")
    return d.set_index("month").apply(_num)


def _boj_prime(path: Path) -> pd.DataFrame:
    """日銀「長・短期プライムレート(主要行)の推移」(HTML)。改定日の表を、各月末時点の適用金利にする。"""
    text = path.read_text(encoding="utf-8")
    rows = []
    for tr in re.findall(r"<tr.*?</tr>", re.search(r"<table.*?</table>", text, re.S).group(0), re.S):
        cells = [html.unescape(re.sub(r"<[^>]+>", "", c)).strip() for c in re.findall(r"<t[hd].*?</t[hd]>", tr, re.S)]
        m = re.search(r"（(\d{4})）年\s*(\d+)月", cells[0]) if cells else None
        if not m or len(cells) < 5:
            continue  # 見出し行など
        vals = [re.match(r"[\d.]+", c) for c in cells[1:5]]  # 短プラ 最頻・最高・最低、長プラ。「↓」は据え置き
        rows.append((f"{m.group(1)}-{int(m.group(2)):02d}", *[float(v.group(0)) if v else np.nan for v in vals]))
    d = pd.DataFrame(rows, columns=["month", "prime_short", "prime_short_max", "prime_short_min", "prime_long"]).ffill()
    d = d.groupby("month").last()  # 同じ月に複数回改定されたら月末の値
    months = pd.period_range(d.index.min(), d.index.max(), freq="M").strftime("%Y-%m")
    return d.reindex(months).ffill()[["prime_short", "prime_long"]]


def _jgb(path: Path) -> pd.Series:
    """財務省「国債金利情報」(日次、和暦)の10年債利回りを月平均にする。"""
    d = pd.read_csv(path, encoding="cp932", skiprows=1, na_values=["-"])
    era = {"S": 1925, "H": 1988, "R": 2018}
    ymd = d["基準日"].str.extract(r"^([SHR])(\d+)\.(\d+)\.(\d+)$")
    month = (ymd[0].map(era) + ymd[1].astype(int)).astype(str) + "-" + ymd[2].astype(int).map("{:02d}".format)
    return _num(d["10年"]).groupby(month).mean().rename("jgb10")


def _rpi(path: Path) -> pd.DataFrame:
    """国交省「不動産価格指数(住宅)」の季節調整済み系列(2010年平均=100)。地域×月の long 形式。"""
    frames = []
    for name, sh in pd.read_excel(path, sheet_name=None, header=None, engine="openpyxl").items():
        if not name.endswith("季節調整"):
            continue
        region = str(sh.iat[0, 11]).strip()
        month = pd.to_datetime(sh[0], errors="coerce")
        d = pd.DataFrame({"region": region, "month": month.dt.strftime("%Y-%m"), "total": _num(sh[1]), "land": _num(sh[4]),
                          "house": _num(sh[7]), "mansion": _num(sh[10])}).dropna(subset=["month"])
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def load_rates() -> tuple[pd.DataFrame, pd.DataFrame]:
    """月次の金利(wide)と不動産価格指数(long)。金利は RATES_FROM 以降、欠けている月は NaN。"""
    raw = RAW / "rates"
    ir01, fm02, ir04 = _boj_csv(raw / "boj_ir01.csv"), _boj_csv(raw / "boj_fm02.csv"), _boj_csv(raw / "boj_ir04.csv")
    col = lambda d, key: d[[c for c in d.columns if key in c][0]]  # noqa: E731
    parts = {
        "policy_rate": ir01.iloc[:, 0],
        "call_rate": col(fm02, "月平均"),
        "lend_new_short": col(ir04, "新規/短期"), "lend_new_long": col(ir04, "新規/長期"),
        "lend_stock_short": col(ir04, "ストック/短期"), "lend_stock_long": col(ir04, "ストック/長期"),
        "jgb10": _jgb(raw / "jgbcm_all.csv"),
    }
    rates = pd.concat({**parts, **dict(_boj_prime(raw / "boj_prime.html").items())}, axis=1)
    rates = rates[rates.index >= RATES_FROM].sort_index().rename_axis("month").reset_index()
    rpi = _rpi(raw / "mlit_rpi.xlsx")
    rpi = rpi[rpi["month"] >= RATES_FROM].sort_values(["region", "month"]).reset_index(drop=True)
    print(f"rates: {rates['month'].iloc[0]}〜{rates['month'].iloc[-1]} ({rates.notna().sum().to_dict()})")
    print(f"rpi: {rpi['region'].nunique()} regions, {rpi['month'].min()}〜{rpi['month'].max()}")
    return rates, rpi


REINS_KIND = {"中古マンション": "mansion", "中古戸建住宅": "house", "新築戸建住宅": "new_house", "土地": "land"}
REINS_STATUS = {"成約": "sold", "新規登録": "new", "在庫": "stock"}
REINS_COLS = {"件数": "n", "m2単価": "unit_price", "価格": "price", "専有面積": "area", "土地面積": "land_area",
              "面積(参考)": "land_area", "建物面積": "floor_area", "築年数": "age"}


def _reins_region(s: str) -> tuple[str, str]:
    """「東京都 城東地区 (台東区、…)」→ (「東京都 城東地区」, 「台東区、…」)。括弧内の市区は合併等で号により変わる。"""
    m = re.match(r"^(.*?)\s*(?:\((.*)\))?$", s)
    return re.sub(r"\s+", " ", m.group(1)).strip(), m.group(2) or ""


def _reins_pages(pages: list[list[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """東日本レインズ「月例速報 Market Watch」(詳細データ PDF)のテキストを、月次の概況と四半期の価格帯別件数にする。

    概況の行: 「25/08 3,553 54.5 84.85 …」(年が変わる月だけ YY/ が付く)。列は見出し「件数 m2単価 価格 …」の順で、
    件数は(値, 前年比)、築年数は値のみ、その他は(値, 前年比, 前月比)が並ぶ。値だけを取る。
    価格帯別: 帯の見出し「~1,000 ~2,000 …」の後に5四半期ぶんの件数行。四半期のラベルは抽出の仕方によって表の外に
    ばらけて出ることがあるため、ページ内のラベルを昇順に並べて行に割り当てる。「別集計」(参考値)は使わない。
    """
    monthly, bands = [], []
    kind = section = status = region = note = cols = year = band_hdr = None
    for page in pages:
        lines = [unicodedata.normalize("NFKC", ln).strip() for ln in page]
        # 成約・新規登録は「2025/04~06」、在庫は四半期末の「2025/06末」
        quarters = sorted({ln[:7] for ln in lines if re.fullmatch(r"\d{4}/\d\d(~\d\d|末)", ln)})
        band_rows: list[tuple] = []
        for ln in lines:
            if m := re.match(r"^[IV]+\.(.+?)(?:\(面積.*\))?レポート", ln):
                kind = REINS_KIND[m.group(1)]
            elif m := re.match(r"^\d\.(.+)", ln):
                section = {"首都圏・都県別概況": "overview", "地域別概況": "overview", "首都圏・都県別価格帯別件数": "band"}.get(m.group(1).strip())
            elif m := re.match(r"^\(\d\)(成約|新規登録|在庫)状況", ln):
                status = REINS_STATUS[m.group(1)]
            elif ln.startswith("○"):
                (region, note), cols, year, band_hdr = _reins_region(ln[1:]), None, None, None
            elif section == "overview" and ln.startswith("件数"):
                cols = [REINS_COLS[c] for c in ln.split()]
            elif section == "overview" and cols and (m := re.match(r"^(?:(\d\d)/)?(\d\d) (.+)$", ln)):
                year = 2000 + int(m.group(1)) if m.group(1) else year
                toks, width = m.group(3).split(), [2 if c == "n" else 1 if c == "age" else 3 for c in cols]
                if len(toks) != sum(width):
                    raise ValueError(f"レインズの表の列数が想定と違います: {kind} {status} {region}: {ln}")
                starts = np.cumsum([0, *width[:-1]])
                vals = {c: pd.to_numeric(toks[i].replace(",", ""), errors="coerce") for c, i in zip(cols, starts)}
                monthly.append({"kind": kind, "status": status, "region": region, "region_note": note,
                                "month": f"{year}-{int(m.group(2)):02d}", **vals})
            elif section == "band" and ln.startswith("~"):
                band_hdr = [t.replace(",", "") for t in ln.split()]
            elif section == "band" and band_hdr and re.fullmatch(r"[\d,]+( [\d,]+)+", ln):
                band_rows.append((kind, status, region, band_hdr, [int(t.replace(",", "")) for t in ln.split()]))
        groups = pd.Series(range(len(band_rows))).groupby([r[2] for r in band_rows], sort=False)
        for idx in groups.indices.values():  # 地域ごとに、行の順 = 四半期の昇順
            if len(idx) != len(quarters):
                raise ValueError(f"レインズの価格帯別の行数が四半期の数と違います: {band_rows[idx[0]][:3]} {len(idx)}行 {quarters}")
            for i, q in zip(idx, quarters):
                k, st, rg, hdr, vals = band_rows[i]
                labels = [*hdr[:-1], f"{hdr[-1]}~", "計"]  # 最後の帯は「10,000万円~」
                if len(vals) != len(labels):
                    raise ValueError(f"レインズの価格帯の数が見出しと違います: {k} {st} {rg} {hdr} {vals}")
                quarter = f"{q[:4]}-Q{(int(q[5:7]) + 2) // 3}"  # 2025/04 も 2025/06 も Q2
                bands += [{"kind": k, "status": st, "region": rg, "quarter": quarter, "band": b, "n": v} for b, v in zip(labels, vals)]
    return pd.DataFrame(monthly), pd.DataFrame(bands)


def load_reins() -> tuple[pd.DataFrame, pd.DataFrame]:
    """data/raw/reins/MW_YYYYMMdata.pdf をすべて読み、号をまたいで重なる月・四半期は新しい号の値を使う。"""
    import pdfplumber  # 文字の座標から行と列を組み立てる(pypdf では一部の号で数字が分解・連結される)

    ms, bs = [], []
    for p in sorted((RAW / "reins").glob("MW_*data.pdf")):
        with pdfplumber.open(p) as pdf:
            m, b = _reins_pages([pg.extract_text().splitlines() for pg in pdf.pages])
        ms.append(m.assign(issue=p.stem[3:9]))
        bs.append(b.assign(issue=p.stem[3:9]))
    m = pd.concat(ms).sort_values("issue").drop_duplicates(["kind", "status", "region", "month"], keep="last")
    b = pd.concat(bs).sort_values("issue").drop_duplicates(["kind", "status", "region", "quarter", "band"], keep="last")
    m["region_note"] = m.groupby("region")["region_note"].transform("last")  # 構成市区は最新号の表記に揃える
    order = list(dict.fromkeys([*ms[-1]["region"], *m["region"]]))  # 地域は最新号の掲載順(首都圏 → 都県 → その内訳)
    m["region"] = pd.Categorical(m["region"], categories=order)
    b["region"] = pd.Categorical(b["region"], categories=order)
    for name, d, col in [("月", m, "month"), ("四半期", b, "quarter")]:
        span = pd.period_range(d[col].min(), d[col].max(), freq="M" if col == "month" else "Q").strftime("%Y-%m" if col == "month" else "%Y-Q%q")
        n = d.groupby(["kind", "status", "region"] + (["band"] if col == "quarter" else []))[col].nunique()
        if (n < len(span)).any():
            raise ValueError(f"レインズの{name}に欠けがあります(号の取得漏れ?): {n[n < len(span)].head().to_dict()}")
    print(f"reins: {m['month'].min()}〜{m['month'].max()} {m['region'].nunique()}地域, 価格帯 {b['quarter'].min()}〜{b['quarter'].max()}")
    return (m.drop(columns="issue").sort_values(["kind", "status", "region", "month"]).reset_index(drop=True),
            b.drop(columns="issue").sort_values(["kind", "status", "region", "quarter"]).reset_index(drop=True))


SALES_INDEX_COLS = {1: "total", 7: "house", 10: "mansion", 13: "mansion_ex30"}  # 各列の次の次がサンプル数


def load_sales_index(path: Path | None = None) -> pd.DataFrame:
    """国交省「既存住宅販売量指数」(月次、季節調整済み、2010年平均=100)と、そのもとの登記件数(季節調整前)。地域×月の long 形式。"""
    frames = []
    for region, sh in pd.read_excel(path or RAW / "market" / "mlit_sales_index.xlsx", sheet_name=None, header=None).items():
        ym = sh[0].astype(str).str.fullmatch(r"\d{6}(\.0)?")  # 下の「年次」の表(4桁)は使わない
        d = sh[ym]
        out = pd.DataFrame({"region": region.strip(), "month": d[0].astype(str).str[:4] + "-" + d[0].astype(str).str[4:6]})
        for c, k in SALES_INDEX_COLS.items():
            out[k] = _num(d[c]).to_numpy()
            out[f"{k}_n"] = _num(d[c + 2]).to_numpy()
        frames.append(out)
    d = pd.concat(frames, ignore_index=True)
    print(f"sales index: {d['region'].nunique()} regions, {d['month'].min()}〜{d['month'].max()}")
    return d


STARTS_COLS = {"総数": "total", "持家": "owner", "貸家": "rental", "給与": "company", "分譲": "sale", "うちマンション": "sale_mansion",
               "うち一戸建": "sale_house"}
ERA = {"令和": 2018, "平成": 1988}


def load_starts(path: Path | None = None) -> pd.DataFrame:
    """国交省「建築着工統計」の都道府県別・利用関係別 新設住宅着工戸数(月ごとのシート)。地域×月の long 形式。

    シートの表題「令和８年８月分 …」から年月を取る。地域は都道府県と、表の下の「合計」(全国)・「首都圏」などの集計行。
    """
    frames = []
    for sh in pd.read_excel(path or RAW / "market" / "starts_pref.xls", sheet_name=None, header=None).values():
        title = unicodedata.normalize("NFKC", " ".join(str(v) for v in sh.iloc[:3].to_numpy().ravel() if isinstance(v, str)))
        m = re.search(r"(令和|平成)\s*(\d+|元)年\s*(\d+)月", title)
        month = f"{ERA[m.group(1)] + (1 if m.group(2) == '元' else int(m.group(2)))}-{int(m.group(3)):02d}"
        head = sh.iloc[2].map(lambda v: unicodedata.normalize("NFKC", v).strip() if isinstance(v, str) else v)
        cols = {STARTS_COLS[h]: i for i, h in head.items() if h in STARTS_COLS and i < 20}  # 右側の別表(コード付き)は使わない
        name = sh[1].map(lambda v: re.sub(r"\s+", "", unicodedata.normalize("NFKC", v)) if isinstance(v, str) else None)
        rows = name.notna() & _num(sh[cols["total"]]).notna()
        d = pd.DataFrame({"region": name[rows].replace({"合計": "全国"}), "month": month,
                          **{k: _num(sh.loc[rows, i]) for k, i in cols.items()}})
        frames.append(d.drop_duplicates("region"))  # 「北海道」「沖縄」は都道府県と地方の両方に出る(値は同じ)
    d = pd.concat(frames, ignore_index=True).sort_values(["region", "month"]).reset_index(drop=True)
    print(f"starts: {d['region'].nunique()} regions, {d['month'].min()}〜{d['month'].max()}")
    return d


CHINTAI_COLS = ["limited_35", "limited_15", "free_35", "free_15"]  # 繰上返済制限あり/なし × 35年/15年固定


def _chintai_lines(lines: list[str]) -> pd.DataFrame:
    """住宅金融支援機構「賃貸住宅融資 参考金利の推移表」の行を月次にする。

    行は「平成18年度 ４月 ― ― 3.02% ―」(年度の最初の月)か「５月 2.65% 2.16% …」。年度の4〜12月はその年、1〜3月は翌年。
    「2.71%（3.72%）」の括弧内はサービス付き高齢者向け住宅(施設共用型)の金利なので使わない。「―」はその型がなかった時期。
    """
    rows, fy = [], None
    for ln in lines:
        ln = unicodedata.normalize("NFKC", ln).strip()
        if m := re.match(r"^(?:\(?(平成|令和)(\d+|元)年度\)?\s*)?(?:\((?:平成|令和)(?:\d+|元)年度\)\s*)?(\d{1,2})月\s+(.+)$", ln):
            if m.group(1):
                fy = ERA[m.group(1)] + (1 if m.group(2) == "元" else int(m.group(2)))
            vals = re.sub(r"\([\d.]+%\)", "", m.group(4)).split()
            if fy is None or len(vals) != 4:
                continue  # 見出しなど
            mo = int(m.group(3))
            rows.append({"month": f"{fy + (mo <= 3)}-{mo:02d}", **{c: _num(pd.Series([v.rstrip("%")])).iloc[0] for c, v in zip(CHINTAI_COLS, vals)}})
    d = pd.DataFrame(rows)
    # 年度の一覧と各年度の月別の表で同じ月が2回出る。値が食い違えば読み違いなので止める
    dup = d.groupby("month")[CHINTAI_COLS].nunique(dropna=False).max(axis=1)
    if (dup > 1).any():
        raise ValueError(f"賃貸住宅融資の金利が同じ月で食い違います: {list(dup[dup > 1].index)}")
    d = d.drop_duplicates("month").sort_values("month").reset_index(drop=True)
    span = pd.period_range(d["month"].iloc[0], d["month"].iloc[-1], freq="M").strftime("%Y-%m")
    if len(span) != len(d):
        raise ValueError(f"賃貸住宅融資の金利に欠けている月があります: {sorted(set(span) - set(d['month']))[:5]}")
    return d


def _boj_api_csv(path: Path) -> pd.DataFrame:
    """日銀 時系列統計データ検索サイト API (getDataCode, format=csv) の四半期系列。列 = 系列コード、index = 「2026-Q1」。"""
    lines = path.read_text(encoding="cp932").splitlines()
    head = next(i for i, ln in enumerate(lines) if ln.startswith("SERIES_CODE,"))
    d = pd.read_csv(io.StringIO("\n".join(lines[head:])), dtype=str)
    d["quarter"] = d["SURVEY_DATES"].str[:4] + "-Q" + d["SURVEY_DATES"].str[5]  # 「202601」= 2026年第1四半期
    return d.pivot(index="quarter", columns="SERIES_CODE", values="VALUES").apply(_num)


BOJ_LOAN = {"DLHLLKG71_DLHL2DSFL": "housing_new", "DLLILKG96_DLLI5DS5THR": "rental_new"}  # 国内銀行の新規貸出(億円)


def load_loan() -> tuple[pd.DataFrame, pd.DataFrame]:
    """機構の賃貸住宅融資 参考金利(月次)と、日銀の住宅資金・個人による貸家業の新規貸出額(四半期)。"""
    import pdfplumber

    with pdfplumber.open(RAW / "loan" / "jhf_chintai_rates.pdf") as pdf:
        chintai = _chintai_lines([ln for pg in pdf.pages for ln in pg.extract_text().splitlines()])
    boj = _boj_api_csv(RAW / "loan" / "boj_la01.csv")[list(BOJ_LOAN)].rename(columns=BOJ_LOAN).rename_axis("quarter").reset_index()
    print(f"chintai: {chintai['month'].iloc[0]}〜{chintai['month'].iloc[-1]}, boj loans: {boj['quarter'].iloc[0]}〜{boj['quarter'].iloc[-1]}")
    return chintai, boj


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
    rates, rpi = load_rates()
    rates.to_parquet(OUT / "rates.parquet", index=False)
    rpi.to_parquet(OUT / "rpi.parquet", index=False)
    reins, bands = load_reins()
    reins.to_parquet(OUT / "reins.parquet", index=False)
    bands.to_parquet(OUT / "reins_bands.parquet", index=False)
    load_sales_index().to_parquet(OUT / "sales_index.parquet", index=False)
    load_starts().to_parquet(OUT / "starts.parquet", index=False)
    detail, areas = load_population_detail()
    detail.to_parquet(OUT / "population_detail.parquet", index=False)
    areas.to_parquet(OUT / "population_areas.parquet", index=False)
    (OUT / "boundaries.json").write_text(json.dumps(load_boundaries(), separators=(",", ":")), encoding="utf-8")
    chintai, boj_loans = load_loan()
    chintai.to_parquet(OUT / "chintai_rates.parquet", index=False)
    boj_loans.to_parquet(OUT / "boj_loans.parquet", index=False)
    print(tx.groupby(["kind", "city"]).size().unstack(0))


if __name__ == "__main__":
    main()
