"""市況・ローンの生データ(レインズ PDF のテキスト / 機構の金利推移表 / 日銀 API / 国交省 Excel)のパース。"""

import pandas as pd
import pytest

from reap import etl

REINS_PAGES = [
    ["I.中古マンションレポート", "1.首都圏・都県別概況", "(1)成約状況", "○首都圏", "件数 m2単価 価格 専有面積 築年数",
     "(件) 前年比(%) (万円) 前年比(%) 前月比(%) (万円) 前年比(%) 前月比(%) (m2) 前年比(%) 前月比(%) (年)",
     "25/12 3,975 25.9 85.08 9.0 3.5 5,340 8.2 2.6 62.77 -0.7 -0.8 27.35",
     "26/01 3,343 3.1 86.99 6.3 2.3 5,493 6.7 2.9 63.15 0.5 0.6 26.78",
     "02 4,241 2.1 85.61 8.2 -1.6 5,458 9.5 -0.7 63.75 1.2 1.0 27.14",
     "○東京都 城東地区 (台東区、江東区)", "件数 m2単価 価格 専有面積 築年数",
     "26/02 502 -10.7 105.25 15.5 1.9 6,405 16.7 4.1 60.85 1.1 2.1 23.46"],
    # 価格帯別: 四半期のラベルが表の外にばらけて出る(行は四半期の昇順)
    ["I.中古マンションレポート", "2.首都圏・都県別価格帯別件数", "(1)成約状況", "○首都圏", "~1,000 ~2,000 2,000", "万円 万円 万円~",
     "10 20 30 60", "( 16.7 ) ( 33.3 ) ( 50.0 ) ( 100.0 )", "11 21 31 63", "年/月 計", "2025/10~12", "2025/07~09", "2025/07~09"],
    ["I.中古マンションレポート", "2.首都圏・都県別価格帯別件数", "(3)在庫状況", "○首都圏", "~1,000 ~2,000 2,000",
     "100 200 300 600", "101 201 301 603", "2025/09末", "2025/12末"],
    # 土地は ㎡単価・価格・面積(参考)で築年数がない。「別集計」(参考値)は読まない
    ["IV.土地(面積100~200m2)レポート", "1.首都圏・都県別概況", "(2)新規登録状況", "○首都圏", "件数 m2単価 価格 面積(参考)",
     "26/02 2,000 1.0 30.00 1.0 1.0 4,000 1.0 1.0 140.00 0.0 0.0",
     "3.〔参考〕別集計 首都圏・都県別概況", "○首都圏", "件数 m2単価 価格 面積(参考)", "26/02 9 1.0 30.00 1.0 1.0 4,000 1.0 1.0 140.00 0.0 0.0"],
]


def test_reins_pages():
    m, b = etl._reins_pages(REINS_PAGES)
    sold = m[(m["kind"] == "mansion") & (m["region"] == "首都圏")]
    assert sold["month"].tolist() == ["2025-12", "2026-01", "2026-02"]  # 年が変わる月だけ YY/ が付く
    assert sold.iloc[1][["n", "unit_price", "price", "area", "age"]].tolist() == [3343, 86.99, 5493, 63.15, 26.78]
    joto = m[m["region"] == "東京都 城東地区"].iloc[0]
    assert (joto["region_note"], joto["month"], joto["n"]) == ("台東区、江東区", "2026-02", 502)
    land = m[m["kind"] == "land"]
    assert len(land) == 1 and land.iloc[0][["status", "n", "land_area"]].tolist() == ["new", 2000, 140.0]
    assert pd.isna(land.iloc[0]["age"])
    s = b[b["status"] == "sold"].pivot(index="quarter", columns="band", values="n")
    assert list(s.index) == ["2025-Q3", "2025-Q4"]
    assert s.loc["2025-Q4", ["~1000", "~2000", "2000~", "計"]].tolist() == [11, 21, 31, 63]
    st = b[b["status"] == "stock"].pivot(index="quarter", columns="band", values="n")
    assert st.loc["2025-Q3", "計"] == 600  # 在庫は「2025/09末」= 7〜9月期


def test_reins_pages_rejects_misread_rows():
    bad = [REINS_PAGES[0][:6] + ["26/01 3,343 3.1 86.99"]]
    with pytest.raises(ValueError, match="列数"):
        etl._reins_pages(bad)
    missing = [[ln for ln in REINS_PAGES[1] if ln != "2025/10~12"]]  # 行は2つあるのに四半期のラベルが1つ
    with pytest.raises(ValueError, match="四半期の数"):
        etl._reins_pages(missing)


def test_chintai_lines():
    lines = ["参 考 金 利 推 移", "平成23年度 ４月 2.79% 2.29% 3.03% 2.54%", "11月 2.71%（3.72%） 2.19%（3.20%） 2.97%（3.98%） 2.50%（3.51%）",
             "平成23年度 ４月 2.79% 2.29% 3.03% 2.54%"]
    lines += [f"{m}月 1.00% 1.00% 1.00% 1.00%" for m in (5, 6, 7, 8, 9, 10)]
    lines += ["12月 2.68% 2.15% 2.93% 2.46%", "１月 2.66% 2.25% 2.91% 2.50%", "（令和元年度） ５月 ― ― 1.63% ―"]
    with pytest.raises(ValueError, match="欠けている月"):  # 令和元年5月だけ離れている
        etl._chintai_lines(lines)
    d = etl._chintai_lines(lines[:-1])
    assert d["month"].tolist()[0] == "2011-04" and d["month"].tolist()[-1] == "2012-01"  # 1〜3月は年度の翌年
    assert d.set_index("month").loc["2011-11"].tolist() == [2.71, 2.19, 2.97, 2.50]  # 括弧内(サ高住)は使わない
    d = etl._chintai_lines(["令和元年度 ４月 ― ― 1.63% ―"])
    assert d.iloc[0]["month"] == "2019-04" and pd.isna(d.iloc[0]["limited_35"])
    with pytest.raises(ValueError, match="食い違い"):
        etl._chintai_lines(["平成23年度 ４月 2.79% 2.29% 3.03% 2.54%", "平成23年度 ４月 2.80% 2.29% 3.03% 2.54%"])


def test_boj_api_csv(tmp_path):
    p = tmp_path / "boj.csv"
    p.write_text("STATUS,200\nMESSAGE,正常に終了しました。\nPARAMETER,DB,LA01\n"
                 "SERIES_CODE,NAME_OF_TIME_SERIES_J,UNIT_J,FREQUENCY,CATEGORY_J,LAST_UPDATE,SURVEY_DATES,VALUES\n"
                 "A,住宅資金,億円,QUARTERLY,x,20260810,202504,42164\nA,住宅資金,億円,QUARTERLY,x,20260810,202601,48813\n"
                 "B,貸家業,億円,QUARTERLY,x,20260810,202601,null\n", encoding="cp932")
    d = etl._boj_api_csv(p)
    assert list(d.index) == ["2025-Q4", "2026-Q1"]
    assert d["A"].tolist() == [42164, 48813] and d["B"].isna().all()


def test_sales_index(tmp_path):
    p = tmp_path / "s.xlsx"
    row = lambda ym, v: [ym, v, 0.1, 100, v, 0.1, 90, v + 1, 0.1, 40, v + 2, 0.1, 60, v + 3, 0.1, 50]  # noqa: E731
    sheet = pd.DataFrame([[None] * 16] * 8 + [row(202601, 120.0), row(202602, 121.0), [None] * 16, ["年次"] + [None] * 15, row(2025, 99.0)])
    with pd.ExcelWriter(p, engine="openpyxl") as w:
        sheet.to_excel(w, sheet_name="東京都", header=False, index=False)
    d = etl.load_sales_index(p)
    assert d["month"].tolist() == ["2026-01", "2026-02"]  # 年次の表は読まない
    assert d.iloc[1][["region", "total", "total_n", "house", "mansion", "mansion_ex30", "mansion_ex30_n"]].tolist() == \
        ["東京都", 121.0, 100, 122.0, 123.0, 124.0, 50]


def test_starts(tmp_path):
    p = tmp_path / "s.xlsx"
    head = [None, None, "総数", None, "持家", None, "貸家", None, "給与", None, "分譲", None, "うちマンション", None, "うち一戸建", None]
    body = [[None, "東　京", 9754, 1, 1177, 1, 5879, 1, 36, 1, 2662, 1, 1111, 1, 1522, 1],
            [None, "合　計", 63974, 1, 17713, 1, 28309, 1, 1114, 1, 16838, 1, 6475, 1, 10182, 1],
            [None, "北海道", 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]] * 1 + [[None, "北海道", 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]]
    with pd.ExcelWriter(p, engine="openpyxl") as w:
        for name, title in [("８年８月", "令和８年８月分　都道府県別"), ("元年５月", "令和元年５月分")]:
            pd.DataFrame([[None] * 16, [None, None, None, title] + [None] * 12, head, [None] * 16, [None] * 16, *body]).to_excel(
                w, sheet_name=name, header=False, index=False)
    d = etl.load_starts(p)
    assert sorted(d["month"].unique()) == ["2019-05", "2026-08"]
    tokyo = d[(d["region"] == "東京") & (d["month"] == "2026-08")].iloc[0]
    assert tokyo[["total", "owner", "rental", "company", "sale", "sale_mansion", "sale_house"]].tolist() == [9754, 1177, 5879, 36, 2662, 1111, 1522]
    assert set(d["region"]) == {"東京", "全国", "北海道"} and len(d) == 6  # 「合計」は全国、重複行は1つに
