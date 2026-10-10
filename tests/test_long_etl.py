"""長期推移タブの生データ(地価公示・消費者物価の都市別 Excel・日銀の API/プライムレートの過去の表)のパースと地価指数。"""

import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from reap import etl


def test_long_city():
    assert etl._long_city("13101") == etl._long_city("13123") == "東京23区"
    assert etl._long_city("14118") == "横浜市" and etl._long_city("14131") == "川崎市"
    assert etl._long_city("27128") == "大阪市" and etl._long_city("27141") == "堺市"
    assert etl._long_city("13201") is None and etl._long_city("14100") is None  # 八王子市、市そのもののコード


def test_koji_points(tmp_path):
    """1983〜1985年の3年分。住宅地(000)だけを市にまとめ、属性移動の1桁目が 1(継続)・2(番号変更)なら継続地点。"""
    def feature(code, use, prices, flags):
        props = {"L01_001": code, "L01_002": use, "L01_061": "111"}
        props.update({f"L01_{62 + i:03d}": p for i, p in enumerate(prices)})
        props.update({f"L01_{65 + i:03d}": f for i, f in enumerate(flags)})
        return {"type": "Feature", "properties": props, "geometry": None}
    feats = [feature("13101", "000", [100, 110, 121], ["10000000000000", "20000000000000"]),
             feature("13110", "000", [0, 200, 220], ["00000000000000", "40000000000000"]),
             feature("13101", "005", [500, 500, 500], ["10000000000000", "10000000000000"]),  # 商業地は除く
             feature("13201", "000", [50, 50, 50], ["10000000000000", "10000000000000"])]  # 23区の外
    p = tmp_path / "L01.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("L01/L01.geojson", json.dumps({"type": "FeatureCollection", "features": feats}))
    d = etl._koji_points(p)
    assert d["city"].tolist() == ["東京23区", "東京23区"]
    assert d["price_1983"].tolist() == [100, 0] and d["price_1985"].tolist() == [121, 220]
    assert d["cont_1984"].tolist() == [True, False] and d["cont_1985"].tolist() == [True, False]


def test_koji_latest(tmp_path):
    """物件の評価タブ用: 住宅地・商業地の地点の最新価格。区は所在地の先頭(都道府県名を外す)が取引データの区名に一致するもの。"""
    def feature(use, addr, price):
        props = {"L01_002": use, "L01_007": 2026, "L01_008": price, "L01_009": 1.5, "L01_025": addr, "L01_048": "駅", "L01_050": 300, "L01_058": 200}
        return {"type": "Feature", "properties": props, "geometry": {"type": "Point", "coordinates": [135.7, 35.0]}}
    feats = [feature("000", "京都府\u3000京都市北区紫野石龍町２９番７", 300000),  # 「京都府」を「京都」+「府」と切らない
             feature("005", "東京都\u3000北区赤羽１丁目８番１０", 5700000),
             feature("009", "東京都\u3000北区志茂２丁目", 400000),  # 工業地は除く
             feature("000", "東京都\u3000八王子市元本郷町", 200000)]  # 取引データにない市
    p = tmp_path / "L01.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("L01/L01.geojson", json.dumps({"type": "FeatureCollection", "features": feats}))
    d = etl.koji_latest(p, ["北区", "京都市北区"])
    assert d["ward"].tolist() == ["京都市北区", "北区"] and d["use"].tolist() == ["住宅地", "商業地"]
    assert d["price"].tolist() == [300000, 5700000] and d["year"].tolist() == [2026, 2026] and d.loc[0, "lat"] == 35.0


def test_koji_index(monkeypatch):
    """継続地点の変化率の対数平均をつなぎ、最新年=100。比べられる地点が少ない年より前は NaN。"""
    monkeypatch.setattr(etl, "KOJI_MIN_POINTS", 2)
    pts = pd.DataFrame({
        "city": ["A"] * 3,
        "price_1983": [100, 0, 0], "price_1984": [110, 100, 0], "price_1985": [121, 121, 300], "price_1986": [121, 121, 300],
        "cont_1984": [True, False, False], "cont_1985": [True, True, False], "cont_1986": [True, True, True],
    })
    d = etl.koji_index(pts).set_index("year")
    assert np.isnan(d.at[1983, "land"])  # 1984年は比べられる地点が1つだけ
    growth = np.exp(np.mean(np.log([121 / 110, 121 / 100])))
    assert d.at[1985, "land"] == pytest.approx(100) and d.at[1984, "land"] == pytest.approx(100 / growth)
    assert d["land_n"].tolist() == [0, 1, 2, 3]
    assert d.at[1986, "land_level"] == pytest.approx((121 + 121 + 300) / 3)
    assert d.at[1984, "land_level"] == pytest.approx(d.at[1986, "land_level"] / growth)


def test_cpi_city(tmp_path):
    p = tmp_path / "cpi.xlsx"
    sheet = pd.DataFrame([[None] * 5 for _ in range(6)])
    sheet.iloc[1, 1] = "大阪市"
    sheet.iloc[2, 2:5] = ["1970年", "1971年", "1972年"]
    sheet.iloc[3, 1:5] = ["1月", 31.6, 33.8, 35.5]
    sheet.iloc[4, 1:5] = ["年平均", 32.5, 34.7, "-"]
    sheet.iloc[5, 1:5] = ["年平均", "*", 6.6, 5.6]  # 前年比の段
    with pd.ExcelWriter(p) as w:
        pd.DataFrame([["シート名", "分類名"], ["A1", "総合"], ["A2", "持家の帰属家賃を除く家賃"]]).to_excel(w, sheet_name="目次", header=False, index=False)
        pd.DataFrame([[1]]).to_excel(w, sheet_name="A1", header=False, index=False)
        sheet.to_excel(w, sheet_name="A2", header=False, index=False)
    s = etl._cpi_city(p)
    assert s.to_dict() == {1970: 32.5, 1971: 34.7}


def test_boj_prime_old_tables(tmp_path):
    """1966〜1988年の表(実施日・短期貸出金利・長プラ)は長プラだけ、1989年以降の表とつなぐ。"""
    old = tmp_path / "old.htm"
    old.write_text("""<table><tr><th>実施日</th><th>短期貸出金利</th><th>長期プライムレート</th></tr>
<tr><td>昭和63（1988）年&nbsp;1月28日</td><td>&darr;</td><td>5.5</td></tr>
<tr><td>昭和63（1988）年&nbsp;8月&nbsp;1日</td><td>&darr;</td><td>5.7</td></tr></table>""", encoding="utf-8")
    new = tmp_path / "new.htm"
    new.write_text("""<table><tr><th>実施日</th><th colspan=3>短期</th><th>長期</th></tr><tr><th>最頻値</th><th>最高値</th><th>最低値</th></tr>
<tr><td>平成 1（1989）年&nbsp;1月23日</td><td>--</td><td>4.25</td><td>--</td><td>5.7 （1988.8.1） 2</td></tr>
<tr><td>平成 1（1989）年&nbsp;1月26日</td><td>4.25</td><td>&darr;</td><td>--</td><td>&darr;</td></tr></table>""", encoding="utf-8")
    d = etl._boj_prime(old, new)
    assert d.index[0] == "1988-01" and d.index[-1] == "1989-01"
    assert d.loc["1988-07", "prime_long"] == 5.5 and d.loc["1988-08", "prime_long"] == 5.7
    assert d["prime_short"].iloc[:-1].isna().all() and d.loc["1989-01", "prime_short"] == 4.25


def test_boj_api_monthly(tmp_path):
    p = tmp_path / "api.csv"
    p.write_text("STATUS,200\nNEXTPOSITION,\nSERIES_CODE,NAME_OF_TIME_SERIES_J,UNIT_J,FREQUENCY,CATEGORY_J,LAST_UPDATE,SURVEY_DATES,VALUES\n"
                 "MADR1M,基準割引率,年％,MONTHLY,基準割引率,20261002,188209,null\n"
                 "MADR1M,基準割引率,年％,MONTHLY,基準割引率,20261002,188210,10.22\n", encoding="cp932")
    assert etl._boj_api_monthly(p).to_dict() == {"1882-10": 10.22}
