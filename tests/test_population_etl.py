"""人口タブの生データ(社人研の推計 Excel のシート / 国土数値情報の境界)の処理。"""

import numpy as np
import pandas as pd

from reap import etl


def test_ipss_sheet_side_by_side():
    """令和5年推計: 男女計・男・女が横に並び、最上位は「90～94歳」「95歳～」。"""
    y = ["2020年", "2050年"]
    rows = [["将来の…"] + [None] * 11, ["13101", "千代田区"] + [None] * 10,
            ["男女計", *y, None, "男", *y, None, "女", *y, None],
            ["総数", 30, 20, None, "総数", 15, 10, None, "総数", 15, 10, None],
            ["0～4歳", 10, 5, None, "0～4歳", 5, 2, None, "0～4歳", 5, 3, None],
            ["90～94歳", 12, 9, None, "90～94歳", 6, 4, None, "90～94歳", 6, 5, None],
            ["95歳～", 8, 6, None, "95歳～", 4, 4, None, "95歳～", 4, 2, None],
            ["（再掲）0～14歳", 10, 5, None, "（再掲）0～14歳", 5, 2, None, "（再掲）0～14歳", 5, 3, None]]
    d = pd.DataFrame(etl._ipss_sheet(pd.DataFrame(rows)), columns=["sex", "year", "age", "pop"])
    assert set(d["sex"]) == {"m", "f"}  # 男女計は使わない(男+女で出す)
    assert d[(d["sex"] == "m") & (d["year"] == 2050)].groupby("age")["pop"].sum().to_dict() == {0: 2, 90: 8}  # 90歳以上にまとまる
    assert len(d) == 2 * 2 * 3  # 再掲・総数は読まない


def test_ipss_sheet_stacked():
    """平成30年推計以前: 男女計・男・女が縦に並び、最上位は「90歳以上」。"""
    rows = [["男女計", "2015年", "2020年"], ["総数", 3, 4], ["0～4歳", 3, 4], [None, None, None],
            ["男", "2015年", "2020年"], ["総数", 1, 2], ["0～4歳", 1, 2], ["90歳以上", 0, 1], [None, None, None],
            ["女", "2015年", "2020年"], ["総数", 2, 2], ["0～4歳", 2, 2]]
    d = pd.DataFrame(etl._ipss_sheet(pd.DataFrame(rows)), columns=["sex", "year", "age", "pop"])
    assert d.set_index(["sex", "year", "age"])["pop"].to_dict() == {
        ("m", 2015, 0): 1, ("m", 2020, 0): 2, ("m", 2015, 90): 0, ("m", 2020, 90): 1, ("f", 2015, 0): 2, ("f", 2020, 0): 2}


def test_simplify():
    line = np.array([[0, 0], [1, 0.0001], [2, 0], [3, 0.0002], [4, 0]], float)
    assert etl._simplify(line, 0.001).tolist() == [[0, 0], [4, 0]]  # ほぼ一直線は端点だけ
    corner = np.array([[0, 0], [1, 0], [2, 0], [2, 1], [2, 2]], float)
    assert etl._simplify(corner, 0.001).tolist() == [[0, 0], [2, 0], [2, 2]]  # 角は残す


def test_census_age(tmp_path):
    """令和7年国勢調査 第2-7表: 男女の行・5歳階級の列。2000年市区町村の組替行(識別コード9)と年齢不詳は使わない。"""
    items = ["00_総数", "01_0～4歳", "19_90～94歳", "20_95～99歳", "21_100歳以上", "22_年齢「不詳」", "R1_（再掲）15歳未満"]
    head = [None] * 9
    rows = [["【不詳補完値】"] + [None] * 15, head + ["人口"] * 7, head + ["年齢"] * 7, head + items, head + [1] * 7, head + ["人"] * 7,
            ["国籍総数か日本人", "男女", "地域識別コード", None, None, None, None, "2025年_地域コード", "地域名"] + [None] * 7]
    def row(nat, sex, kind, code, vals):  # noqa: E306
        return [nat, sex, kind, None, None, 2000, None, code, "x"] + vals
    v = [100, 10, 3, 2, 1, 5, 10]
    rows += [row("0_国籍総数", s, k, c, v) for s in ("0_総数", "1_男", "2_女") for k, c in [("a", "00000"), ("3", "07204"), ("9", "07999")]]
    rows += [row("1_うち日本人", "1_男", "a", "00000", [1] * 7)]
    rows += [row("0_国籍総数", s, "3", c, v) for s in ("1_男", "2_女") for c in etl.HAMADORI[1:]]
    p = tmp_path / "c.xlsx"
    pd.DataFrame(rows).to_excel(p, header=False, index=False)
    d = etl.load_census_age(p)
    nat = d[(d["code"] == "00000") & (d["sex"] == "m")].set_index("age")["population"].to_dict()
    assert nat == {0: 10, 90: 6} and set(d["year"]) == {2025}  # 90歳以上をまとめ、不詳・再掲・日本人は使わない
    hama = d[(d["code"] == "07999") & (d["sex"] == "f")].set_index("age")["population"].to_dict()
    assert hama == {0: 10 * 13, 90: 6 * 13}  # 浜通り13市町村の合計(識別コード9の行は使わない)


def test_vacancy_tables_2023_style(tmp_path):
    """令和5年・平成30年: 地域は「13101_千代田区」、空き家の種類は列の見出し、所有の関係は行(「2_借家」)。"""
    vac = [[None, "項目名", "0_総数", "221_賃貸・売却用及び二次的住宅を除く空き家", "222_賃貸用の空き家"],
           ["a", "00000_全国", 1000, 50, 200], ["0", "13101_千代田区", 100, "-", 20]]
    ten = [[None, "地域区分", "住宅の所有の関係", None], ["a", "00000_全国", "0_総数", 800], ["a", "00000_全国", "2_借家", 300],
           ["0", "13101_千代田区", "1_持ち家", 30], ["0", "13101_千代田区", "2_借家", 60]]
    pd.DataFrame(vac).to_excel(tmp_path / "v.xlsx", header=False, index=False)
    pd.DataFrame(ten).to_excel(tmp_path / "t.xlsx", header=False, index=False)
    assert etl._vacancy_table(tmp_path / "v.xlsx").to_dict() == {"00000": 200, "13101": 20}
    assert etl._tenure_table(tmp_path / "t.xlsx").to_dict() == {"00000": 300, "13101": 60}


def test_vacancy_tables_2013_style(tmp_path):
    """平成25年(都道府県ごと): 地域は7桁の数値(地域コード+2桁)、借家の行のラベルは全角空白入りで、数値の前に注記の列がある。"""
    vac = [[None, None, None, "二次的住宅", "賃貸用の住宅"], [2.0, 1310036.0, "特別区部", 5, 400], [2.0, 1310156.0, "千代田区", 1, 30]]
    ten = [[None, None, "住宅の種類", None, None], [4.0, 1310156.0, "住\u3000宅\u3000総\u3000数", "1)", 100],
           [4.0, 1310156.0, "\u3000借\u3000\u3000\u3000家", None, 70], [4.0, 110036.0, "借家", None, 9]]
    pd.DataFrame(vac).to_excel(tmp_path / "v.xlsx", header=False, index=False)
    pd.DataFrame(ten).to_excel(tmp_path / "t.xlsx", header=False, index=False)
    assert etl._vacancy_table(tmp_path / "v.xlsx").to_dict() == {"13100": 400, "13101": 30}
    assert etl._tenure_table(tmp_path / "t.xlsx").to_dict() == {"13101": 70, "01100": 9}


def test_vacancy_timeseries(tmp_path):
    """時系列統計表: 見出しの列の塊(次の見出しまで)を 地域×年 に。都道府県は2桁、大都市は5桁。"""
    rows = [[None, None, "総数", None, "賃貸用の空き家", None, "売却用の空き家"],
            [None, "Area", "1983", "2023", "1983", "2023", "2023"],
            [None, "00", 10, 20, 3, 4, 1], [None, "13", 5, 6, "-", 2, 1], [None, "13100", 4, 5, 1, 2, 0]]
    pd.DataFrame(rows).to_excel(tmp_path / "ts.xlsx", header=False, index=False)
    d = etl._vacancy_timeseries(tmp_path / "ts.xlsx", "賃貸用の空き家", "売却用")
    assert d.set_index(["code", "year"])["value"].to_dict() == {
        ("00000", 1983): 3, ("00000", 2023): 4, ("13000", 2023): 2, ("13100", 1983): 1, ("13100", 2023): 2}
