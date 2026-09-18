// 画面から呼ぶ分析処理(docs/js/service.js)のテスト。データは pytest が SITE_DIR に書き出した合成データ。
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import test from 'node:test';
import { setLoader } from '../../docs/js/data.js';
import { routes } from '../../docs/js/service.js';

setLoader(async p => { const b = await readFile(path.join(process.env.SITE_DIR, p)); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); });
const FIT = { city: 'X市', kind: 'mansion' };
const close = (a, b, tol = 1e-9) => assert.ok(Math.abs(a - b) <= tol * Math.max(1, Math.abs(b)), `${a} ≠ ${b}`);

test('fit: 誤差の指標、予測と実績の標本、検証用データでの誤差', async () => {
  const r = await routes.fit(FIT);
  assert.equal(r.n, 3000);
  assert.ok(r.r2 > 0.7 && r.r2 < 1 && r.rmse > 0.18 && r.rmse < 0.23);  // 合成データの個別誤差 sd=0.2
  assert.equal(r.pred_actual.pred.length, 3000);
  assert.ok(r.holdout.rmse < 0.25 && r.holdout.n_test > 400);
  assert.ok(r.within_1sd > 0.6 && r.within_1sd < 0.8);
  assert.deepEqual(Object.keys(r.categoricals), ['structure', 'renovated']);
});

test('fit: 切片がそのまま基準物件の価格になる', async () => {
  const r = await routes.fit(FIT), b = r.base;
  close(b.price, Math.exp(r.coefficients.find(c => c.name === 'const').coef));
  const property = { ward: b.ward, age: b.age, area: b.area, station_min: b.station_min, year: b.year,
    ...Object.fromEntries(Object.entries(r.categoricals).map(([k, v]) => [k, v.levels[0]])) };
  const p = await routes.profile({ fit: FIT, vary: 'age', property });
  close(p.price[b.age], b.price);
});

test('profile: 築年数で下がり、帯が推定を挟む。地区を選ぶと地区効果ぶん動く', async () => {
  const plain = await routes.profile({ fit: FIT, vary: 'age', property: { ward: 'B区', area: 60 } });
  assert.ok(plain.price[0] > plain.price.at(-1) && plain.n_points > 0 && plain.fixed.ward === 'B区');
  plain.price.forEach((m, i) => assert.ok(plain.pi_low[i] < plain.ci_low[i] && plain.ci_low[i] < m && m < plain.ci_high[i] && plain.ci_high[i] < plain.pi_high[i]));
  const ds = await routes.districts({ fit: FIT, ward: 'B区' }), best = ds.reduce((a, b) => b.effect > a.effect ? b : a);
  const chosen = await routes.profile({ fit: FIT, vary: 'age', property: { ward: 'B区', area: 60, district: best.district } });
  close(chosen.price[0], plain.price[0] * Math.exp(best.effect));
  close(chosen.district_effect, best.effect);
});

test('forecast: 経年減価と人口効果の掛け合わせ', async () => {
  const r = await routes.forecast({ fit: FIT, property: { ward: 'B区', age: 10 } });
  assert.deepEqual([r.years[0], r.years.at(-1)], [2020, 2050]);
  close(r.combined[0], r.base_price);
  close(r.combined.at(-1), r.age_only.at(-1) * r.pop_only.at(-1) / r.base_price);
  assert.ok(r.combined_low.at(-1) < r.combined.at(-1) && r.combined.at(-1) < r.combined_high.at(-1));
});

test('yield: 統計の家賃×築年補正と、前提の上書き', async () => {
  const body = { fit: FIT, property: { ward: 'B区', age: 20, area: 25 } };
  const r = await routes.yield(body);
  close(r.rent, 60000);  // 築20年は補正係数 1.0
  close(r.gross, 60000 * 12 / r.price);
  close(r.net, (60000 * 12 * 0.95 * 0.95 - 12000 * 12 - 50000) / (r.price * 1.07));
  assert.ok(r.curve.gross[40] > r.curve.gross[0]);  // 価格の下落が家賃より速い
  assert.deepEqual(r.wards.map(w => w.ward).sort(), ['A区', 'B区']);  // 家賃統計のない C区 は除外
  const o = await routes.yield({ ...body, costs: { rent_month: 80000, vacancy_pct: 0 } });
  assert.ok(o.rent === 80000 && o.rent_overridden && o.net > r.net);
  await assert.rejects(routes.yield({ ...body, costs: { bogus: 1 } }), /前提が不正/);
});

test('district-map: 区効果+地区効果', async () => {
  const [m, r] = await Promise.all([routes['district-map'](FIT), routes.fit(FIT)]);
  assert.deepEqual([m.lat.length, m.n_missing], [16, 8]);  // C区の8地区は位置不明
  const i = m.ward.findIndex(w => w !== m.base_ward);
  const ward = r.ward_effects.find(w => w.name === `ward=${m.ward[i]}`).coef;
  const d = (await routes.districts({ fit: FIT, ward: m.ward[i] })).find(x => x.district === m.district[i]).effect;
  close(m.effect[i], ward + d);
});

test('market: 推移は全期間、一覧は取引年で絞る。利回りは標準物件で揃える', async () => {
  const q = { region: 'X市', area_min: 40, area_max: 60, age_min: 10, age_max: 30, station_max: 30, year_from: 2019, year_to: 2020 };
  const r = await routes.market(q), row = r.rows[0];
  assert.equal(r.by, 'ward');
  assert.deepEqual(Object.keys(row.by_year).map(Number).sort(), [2015, 2016, 2017, 2018, 2019, 2020]);
  assert.equal(row.n, row.n_by_year[2019] + row.n_by_year[2020]);
  assert.deepEqual(r.standard, { age: 20, area: 50, station_min: 10, year: 2020 });
  for (const x of r.rows.filter(x => x.std_price)) {
    const p = await routes.profile({ fit: FIT, vary: 'age', property: { ...r.standard, ward: x.name } });
    close(x.std_price, p.price[20]); close(x.std_gross, x.std_rent * 12 / x.std_price);
    assert.ok(!('gross' in x) && !('rent' in x));
  }
  const all = await routes.market({ area_min: 20, area_max: 100, station_max: 30, year_from: 2015 });
  assert.deepEqual([all.by, all.rows.map(x => x.name).sort()], ['city', ['X市', 'Y市']]);
  await assert.rejects(routes.market({ region: 'X市', area_min: 1, area_max: 2 }), /条件に合う取引がありません/);
});

test('全都市プール: 基準の区との差で区効果を返す', async () => {
  const r = await routes.fit({ city: '全都市', kind: 'mansion' });
  assert.ok(r.n === 6000 && r.holdout === null && r.base_year === null && r.year_effects.length === 0);
  assert.equal(r.ward_effects.length, 5);  // 6区のうち基準の区を除く
  close(r.base.price, Math.exp(0) * r.base.price);
});

test('取引が少なすぎる条件はエラー', async () => {
  await assert.rejects(routes.fit({ ...FIT, year_from: 2049 }), /少なすぎます/);
});
