"""金利・不動産価格指数の生データ(日銀 CSV / プライムレート HTML / 財務省 CSV / 国交省 Excel)のパース。"""

import pandas as pd
import pytest

from reap import etl


def test_boj_csv(tmp_path):
    p = tmp_path / "boj.csv"
    p.write_text('主要時系列統計データ表\n2026/09/18 15:00\n"","コールレート（月次）","コールレート（月次）"\n'
                 '"系列名称","無担保コールレート・Ｏ／Ｎ　月末／金利","無担保コールレート・Ｏ／Ｎ　月平均／金利"\n'
                 '"データコード",FM02\'A,FM02\'B\n"単位",年％,年％\n"収録開始期","1985/07","1985/07"\n"収録終了期","2026/08","2026/08"\n'
                 "2026/07,1.0,0.978\n2026/08,ND,0.977\n", encoding="cp932")
    d = etl._boj_csv(p)
    assert list(d.index) == ["2026-07", "2026-08"]
    assert d["無担保コールレート・Ｏ／Ｎ　月平均／金利"].tolist() == [0.978, 0.977]
    assert pd.isna(d.iloc[1, 0])


def test_boj_prime(tmp_path):
    p = tmp_path / "prime.htm"
    p.write_text("""<table><tr><th>実施日</th><th colspan=3>短期</th><th>長期</th></tr><tr><th>最頻値</th><th>最高値</th><th>最低値</th></tr>
<tr><td>平成13（2001）年&nbsp;2月&nbsp;9日</td><td>1.500（2000年&nbsp;8月24日）4</td><td>1.625</td><td>1.500</td><td>2.05</td></tr>
<tr><td>平成13（2001）年&nbsp;3月&nbsp;9日</td><td>&darr;</td><td>&darr;</td><td>&darr;</td><td>1.9</td></tr>
<tr><td>平成13（2001）年&nbsp;3月28日</td><td>1.375</td><td>&darr;</td><td>&darr;</td><td>&darr;</td></tr>
<tr><td>平成13（2001）年&nbsp;6月&nbsp;1日</td><td>&darr;</td><td>&darr;</td><td>&darr;</td><td>1.7</td></tr></table>""", encoding="utf-8")
    d = etl._boj_prime(p)
    assert list(d.index) == ["2001-02", "2001-03", "2001-04", "2001-05", "2001-06"]
    assert d["prime_short"].tolist() == [1.5, 1.375, 1.375, 1.375, 1.375]  # 3月は月内2回の改定の後の値、4〜5月は据え置き
    assert d["prime_long"].tolist() == [2.05, 1.9, 1.9, 1.9, 1.7]


def test_jgb(tmp_path):
    p = tmp_path / "jgb.csv"
    p.write_text("国債金利情報,,,(単位 : %)\n基準日,1年,2年,10年\nS49.9.24,10.3,9.3,-\nH1.1.10,4.0,4.1,4.9\nR8.8.28,1.4,1.7,2.9\nR8.8.31,1.5,1.7,3.0\n",
                 encoding="cp932")
    s = etl._jgb(p)
    assert s.index.tolist() == ["1974-09", "1989-01", "2026-08"]
    assert s.tolist()[1:] == [4.9, 2.95] and pd.isna(s.iloc[0])


def test_rpi(tmp_path):
    p = tmp_path / "rpi.xlsx"
    sheet = pd.DataFrame([[None] * 10 + ["1", "全国"]] + [[None] * 12] * 8
                         + [[pd.Timestamp("2010-06-01"), 100.5, 0.1, 10, 99.0, 0, 5, 98.0, 0, 3, 101.0, 0]]
                         + [[pd.Timestamp("2025-12-01"), 148.0, 0.5, 10, 119.8, 0, 5, 121.9, 0, 3, 225.1, 0]]
                         + [["（注）出典を明示すること"] + [None] * 11])
    with pd.ExcelWriter(p, engine="openpyxl") as w:
        sheet.to_excel(w, sheet_name="全国Japan季節調整", header=False, index=False)
        sheet.to_excel(w, sheet_name="全国Japan原系列", header=False, index=False)  # 原系列は使わない
    d = etl._rpi(p)
    assert d["region"].unique().tolist() == ["全国"] and d["month"].tolist() == ["2010-06", "2025-12"]
    assert d[["total", "land", "house", "mansion"]].iloc[1].tolist() == pytest.approx([148.0, 119.8, 121.9, 225.1])
