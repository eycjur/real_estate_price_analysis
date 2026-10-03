"""取引の種別の振り分け: 土地と建物のうち用途が共同住宅だけのものは一棟(構造で RC系/木造系)、マンション・戸建は住宅用途だけ。構造は4区分にまとめる。"""

import numpy as np
import pandas as pd

from reap import etl


def test_clean_classifies_buildings():
    rows = [  # (種別, 用途, 構造, 面積(マンションは専有・それ以外は土地), 延床)
        ("mansion", None, "RC", 60, np.nan), ("mansion", "事務所", "RC", 60, np.nan),
        ("mansion", "住宅", "SRC、RC", 60, np.nan), ("mansion", "住宅", None, 60, np.nan),
        ("mansion", "住宅", "軽量鉄骨造", 60, np.nan), ("house", "住宅", "ブロック造、木造", 120, 100),
        ("house", "住宅", "木造", 120, 100), ("house", "住宅、店舗", "木造", 120, 100),
        ("house", "共同住宅", "RC", 300, 600), ("house", "共同住宅", "SRC", 300, 900), ("house", "共同住宅", "鉄骨造", 200, 250),
        ("house", "共同住宅", "RC、木造", 200, 250), ("house", "共同住宅、店舗", "RC", 300, 600),
    ]
    df = pd.DataFrame(rows, columns=["kind", "use", "structure", "area", "floor_area"])
    df = df.assign(ward_code=13101, ward="千代田区", district="丸の内", price=5e7, built_year=2000, year=2020, station_min=5, remarks=None)
    out = etl.clean(pd.concat([df] * 300, ignore_index=True))  # 都市×年の外れ値除去(上下0.5%)で落ちないよう件数を増やす
    got = out.groupby(["kind", "structure"]).size().to_dict()
    # 構造は SRC・RC・鉄骨造(軽量鉄骨・ブロック造を含む)・木造。混構造は強いほう、未記載は「不明」
    assert got == {("mansion", "RC"): 300, ("mansion", "SRC"): 300, ("mansion", "不明"): 300, ("mansion", "鉄骨造"): 300,
                   ("house", "木造"): 300, ("house", "鉄骨造"): 300, ("bldg_rc", "RC"): 300, ("bldg_rc", "SRC"): 300, ("bldg_wood", "鉄骨造"): 300}
    b = out[out["kind"] == "bldg_rc"].iloc[0]
    assert (b["area"], b["land_area"]) == (600, 300)  # 一棟は延床を規模、土地面積も持つ
    assert np.isnan(out[out["kind"] == "mansion"].iloc[0]["land_area"])
