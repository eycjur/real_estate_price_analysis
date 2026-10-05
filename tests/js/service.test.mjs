// 画面から呼ぶ分析処理(docs/js/service.js)のテスト。データは pytest が SITE_DIR に書き出した合成データ。
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import test from 'node:test';
import { setLoader } from '../../docs/js/data.js';
import { bandLabels, routes } from '../../docs/js/service.js';

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

test('fit/profile: 面積は5㎡刻みの区分。基準の区分と、区分の中央を並べた断面', async () => {
  const r = await routes.fit(FIT);
  assert.equal(r.area_step, 5); assert.equal(r.base.area, 60);
  assert.ok(r.coefficients.some(c => c.name === 'area=20') && !r.coefficients.some(c => c.name === 'ln_area'));
  const p = await routes.profile({ fit: FIT, vary: 'area', property: { ward: 'B区', age: 10 } });
  assert.ok(p.x.every(x => x % 5 === 2.5) && p.x.every((x, i) => i === 0 || x - p.x[i - 1] === 5));
  assert.ok(p.price.at(-1) > p.price[0]);  // 合成データは面積とともに上がる
  const cats = Object.fromEntries(Object.entries(r.categoricals).map(([k, v]) => [k, v.levels[0]]));  // 未指定だと面積の近い取引の最頻値になる
  const at = async area => (await routes.profile({ fit: FIT, vary: 'age', property: { ward: 'B区', area, ...cats } })).price[10];
  close(await at(60), await at(64)); assert.ok(await at(65) !== await at(64));  // 区分の中は同じ推定
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

test('yield: 統計の家賃×築年補正、ローン返済後の利回り、前提の上書き', async () => {
  const body = { fit: FIT, property: { ward: 'B区', age: 20, area: 25 } };
  const r = await routes.yield(body);
  close(r.rent, 60000);  // 築20年は補正係数 1.0
  close(r.gross, 60000 * 12 / r.price);
  const noi = 60000 * 12 * 0.95 * 0.95 - 12000 * 12 - 50000;
  close(r.net, noi / (r.price * 1.07));
  const pay = (P, pct, n) => { const q = pct / 100 / 12; return P * q / (1 - Math.pow(1 + q, -n * 12)); };
  close(r.rate, 3.0);  // 既定は短期プライムレートの年平均(2020年 = 1.5%) +1.5%
  close(r.payment, pay(r.price, 3.0, 35)); close(r.cash, (noi - r.payment * 12) / (r.price * 1.07));
  assert.ok(r.curve.gross[40] > r.curve.gross[0]);  // 価格の下落が家賃より速い
  close(r.curve.cash[20], r.cash); close(r.curve.payment[0], pay(r.curve.price[0], 3.0, 35));  // 築年数の断面は同じ金利
  assert.deepEqual(r.wards.map(w => w.ward).sort(), ['A区', 'B区']);  // 家賃統計のない C区 は除外
  assert.ok(r.wards.every(w => w.cash != null));
  // 購入した年の断面: 各年の推定価格と年平均の金利、家賃は固定
  const b = r.by_year;
  assert.deepEqual(b.year, [2015, 2016, 2017, 2018, 2019, 2020]);
  assert.deepEqual(b.rate.map(v => +v.toFixed(6)), [2.5, 2.6, 2.7, 2.8, 2.9, 3]);
  const prof = await routes.profile({ ...body, vary: 'year' });
  prof.x.forEach((y, i) => close(b.price[b.year.indexOf(y)], prof.price[i]));  // 取引年の断面と同じ推定価格
  b.year.forEach((_, i) => { close(b.payment[i], pay(b.price[i], b.rate[i], 35)); close(b.gross[i], 60000 * 12 / b.price[i]); close(b.cash[i], (noi - b.payment[i] * 12) / (b.price[i] * 1.07)); });
  close(b.net[5], r.net); close(b.cash[5], r.cash);  // 最新年 = 選択中の取引年
  const o = await routes.yield({ ...body, costs: { rent_month: 80000, vacancy_pct: 0 }, loan: { rate: 'prime_short', spread_pct: -0.5, years: 20 } });
  assert.ok(o.rent === 80000 && o.rent_overridden && o.net > r.net);
  close(o.rate, 1.0); close(o.payment, pay(o.price, 1.0, 20)); assert.equal(o.loan.years, 20);
  const none = await routes.yield({ ...body, loan: { rate: 'jgb10' } });  // 合成データでは値がない系列
  assert.ok(none.rate == null && none.cash == null && none.by_year.cash.every(v => v == null) && none.net === r.net);
  await assert.rejects(routes.yield({ ...body, costs: { bogus: 1 } }), /前提が不正/);
  await assert.rejects(routes.yield({ ...body, loan: { rate: 'bogus' } }), /系列が不正/);
});

test('rates: 金利・価格指数の月次系列と、各年の推定価格の指数', async () => {
  const body = { fit: FIT, property: { ward: 'B区', age: 20, area: 25 } };
  const r = await routes.rates(body);
  assert.deepEqual(r.years, [2015, 2016, 2017, 2018, 2019, 2020]);
  const prof = await routes.profile({ ...body, vary: 'year' });
  prof.x.forEach((y, i) => close(r.price[r.years.indexOf(y)], prof.price[i]));  // 取引年の断面と同じ推定価格
  close(r.index[0], 100); close(r.index[1], r.price[1] / r.price[0] * 100);  // 2010年がないので最初の年=100
  assert.equal(r.index_base_year, 2015);
  assert.equal(r.region, '全国'); assert.equal(r.rpi.mansion.length, r.month.length); assert.equal(r.rates.prime_short.length, r.month.length);
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

test('housing-market: 成約率・在庫月数は12か月合計、価格帯は直近4四半期、地域の比較', async () => {
  const r = await routes['housing-market']({ kind: 'mansion', region: '首都圏' }), L = r.latest, m = r.monthly;
  assert.equal(r.price_key, 'unit_price');
  assert.equal(m.contract_rate12[10], null);  // 12か月そろうまでは出さない
  close(m.contract_rate12[11], 0.25); close(m.months_of_stock12[23], 12); close(m.contract_rate[0], 0.25);
  assert.deepEqual([m.freq, L.period, L.sold, L.new, L.stock], ['M', '2025-12', 100, 400, 1200]);
  close(L.sold_yoy, 0); close(L.price, 73); close(L.price_yoy, 73 / 61 - 1); close(L.ask_gap, 83 / 73 - 1);
  const b = r.bands;
  assert.deepEqual(b.recent, { from: '2025-Q1', to: '2025-Q4' });
  assert.deepEqual(b.rows.map(x => [x.sold, x.new, x.stock]), [[40, 160, 100], [80, 160, 50]]);
  close(b.rows[0].contract_rate, 0.25); close(b.rows[0].months_of_stock, 100 / (40 / 12)); close(b.share[0][0], 1 / 3);
  const j = r.regions.find(x => x.region === '東京都 城東地区');
  close(j.contract_rate, 0.5); close(j.months_of_stock, 6); close(j.price, 80 + 17.5);  // 直近12か月(i=12〜23)の件数加重平均
  assert.deepEqual(r.regions_by_kind, { mansion: ['首都圏', '東京都 城東地区'], land: ['首都圏'] });
  const land = await routes['housing-market']({ kind: 'land', region: '首都圏' });
  assert.ok(land.price_key === 'price' && land.bands === null);
  await assert.rejects(routes['housing-market']({ kind: 'land', region: '東京都 城東地区' }), /データがありません/);
  assert.deepEqual(r.band_regions, ['首都圏']);
});

test('housing-market: 価格帯で絞り込むと、件数・成約率・在庫月数がその価格帯の四半期の値になる', async () => {
  const r = await routes['housing-market']({ kind: 'mansion', region: '首都圏', band: '~1000' }), m = r.monthly, L = r.latest;
  assert.deepEqual([m.freq, m.period.length, L.period, L.sold, L.new, L.stock, r.band_name], ['Q', 8, '2025-Q4', 10, 40, 100, '1,000万円以下']);
  assert.equal(m.contract_rate12[2], null);  // 4四半期そろうまでは出さない
  close(m.contract_rate12[3], 0.25); close(m.months_of_stock12[7], 100 / (40 / 12)); close(m.months_of_stock[0], 100 / (10 / 3));
  close(m.sold_avg[7], 10);  // 4四半期の平均(1四半期あたり)
  close(L.price, 73); assert.equal(m.month.length, 24);  // 価格の推移は全価格帯の月次のまま
  await assert.rejects(routes['housing-market']({ kind: 'mansion', region: '東京都 城東地区', band: '~1000' }), /首都圏と都県だけ/);
  await assert.rejects(routes['housing-market']({ kind: 'mansion', region: '首都圏', band: '~5' }), /価格帯が不正/);
});

test('bandLabels: 万円の区切りを範囲の表示名にし、1億以上は億で書く', () => {
  assert.deepEqual(bandLabels(['~1000', '~2000', '~10000', '~20000', '20000~']),
    ['1,000万円以下', '1,000万〜2,000万円', '2,000万〜1億円', '1億〜2億円', '2億円超']);
  assert.deepEqual(bandLabels(['~7000', '~10000', '10000~']).slice(1), ['7,000万〜1億円', '1億円超']);
});

test('loan-calc: 元利均等は毎月同額、元金均等は元金が一定。利息の合計 = 総返済額 − 借入額', () => {
  const a = routes['loan-calc']({ principal: 1e7, rate: 1.2, years: 10 });
  const q = 0.012 / 12, pay = 1e7 * q / (1 - Math.pow(1 + q, -120));
  close(a.first.payment, pay); close(a.first.interest, 1e7 * q); close(a.first.principal, pay - 1e7 * q); close(a.last_payment, pay);
  close(a.total_payment, pay * 120); close(a.total_interest, pay * 120 - 1e7);
  assert.equal(a.yearly.year.length, 10); close(a.yearly.balance.at(-1), 0, 1e-6);
  close(a.yearly.principal.reduce((x, y) => x + y), 1e7); close(a.yearly.interest.reduce((x, y) => x + y), a.total_interest);
  const p = routes['loan-calc']({ principal: 1.2e6, rate: 1.2, years: 10, method: 'principal' });
  close(p.first.principal, 1e4); close(p.first.interest, 1.2e6 * q); close(p.last_payment, 1e4 + 1e4 * q);
  close(p.total_interest, q * 1e4 * (120 * 121 / 2));  // 毎月の残高 120万, 119万, … 1万 の利息の合計
  assert.ok(p.total_interest < routes['loan-calc']({ principal: 1.2e6, rate: 1.2, years: 10 }).total_interest);
  close(routes['loan-calc']({ principal: 1.2e6, rate: 0, years: 10 }).first.payment, 1e4);
  assert.throws(() => routes['loan-calc']({ principal: 1e6, rate: 1, years: 10, method: 'x' }), /返済方式/);
  assert.throws(() => routes['loan-calc']({ principal: -1, rate: 1, years: 10 }), /借入額/);
});

test('cashflow: 購入時・ランニング・入れ替えのコストと、売却する年数×価格ごとの総収支・IRR', () => {
  const q = { price: 2e7, rent_month: 8e4, rate: 0, years: 20, down_pct: 10, brokerage: false, sale_brokerage: true,
    purchase: { loan_fee_pct: 2.2, scrivener: 15e4, registration_pct: 1.2, stamp: 3e4, acquisition_tax_pct: 0.7, fire_insurance: 3e4, settlement: 5e4 },
    running: { management: 9000, repair_reserve: 9000, rental_mgmt_pct: 5, property_tax: 5000, equipment: 3000, insurance: 1000, accountant: 0 },
    turnover: { interval_years: 4, restoration: 10e4, move_out: 2e4, key: 2e4, utilities_month: 3000, ad_months: 1, agent_months: 0.5, free_rent_months: 0.5, vacancy_months: 2 } };
  const r = routes.cashflow(q);
  assert.deepEqual([r.hold_years, r.price_changes], [[5, 10, 15, 20, 30], [-30, -20, -10, 0, 10, 20]]);
  close(r.loan_amount, 1.8e7);
  close(r.initial.costs, 103.6e4); close(r.initial.total, 303.6e4);  // 事務手数料39.6万+登録免許税24万+取得税14万+定額26万、頭金200万
  close(r.running.month, 31000);  // 管理手数料は家賃の5%=4000円
  close(r.turnover.per_turnover, 46.6e4); close(r.turnover.year, 11.65e4);  // 定額14万+光熱費0.6万+家賃4か月分32万、4年に1回
  const Y = r.yearly;
  close(Y.principal[0] + Y.interest[0], 90e4); close(Y.interest[0], 0);  // 金利0: 1800万 ÷ 20年
  close(Y.cf[0], 96e4 - 37.2e4 - 11.65e4 - 90e4);
  const g = r.grid[1][3];  // 10年後に購入価格で売る: 売却手数料72.6万、残債900万
  close(g.sale_cost, 72.6e4); close(g.balance, 900e4); close(g.total, -303.6e4 + 10 * -42.85e4 + (2000e4 - 72.6e4 - 900e4));
  close(g.total, g.total_base);  // リスクなし
  const flows = [-303.6e4, ...Array(9).fill(-42.85e4), -42.85e4 + 2000e4 - 72.6e4 - 900e4];
  close(flows.reduce((s, v, t) => s + v / Math.pow(1 + g.irr, t), 0), 0, 1e-6);
  // 収入(家賃+売却価格) − 支出(自己資金・費用・ローン・売却時の費用と残債) = 総収支
  close(g.rent + g.sale_price - r.initial.total - g.management - g.repair - g.rental_mgmt - g.other - g.turnover - g.interest - g.principal - g.sale_cost - g.balance, g.total, 1e-6);
  const g30 = r.grid[4][3];  // 返済期間(20年)を過ぎると残債0・返済なし
  close(g30.balance, 0); close(g30.cash_flow, 20 * -42.85e4 + 10 * 47.15e4);
  assert.equal(r.grid[0][0].irr, null);  // 5年後に3割安で売ると売却代金が残債に届かず、最後まで支出だけ
  close(routes.cashflow({ ...q, brokerage: true }).initial.costs, 103.6e4 + 72.6e4);
  assert.equal(routes.cashflow({ ...q, down_pct: 100 }).yearly.principal[0], 0);
  assert.throws(() => routes.cashflow({ ...q, running: { ...q.running, management: -1 } }), /management/);
});

test('cashflow: 金利上昇・家賃下落・修繕積立金の上昇', () => {
  const q = { price: 2e7, rent_month: 1e5, rate: 1, years: 20, down_pct: 0, brokerage: false, sale_brokerage: false,
    purchase: { loan_fee_pct: 0, scrivener: 0, registration_pct: 0, stamp: 0, acquisition_tax_pct: 0, fire_insurance: 0, settlement: 0 },
    running: { management: 0, repair_reserve: 1e4, rental_mgmt_pct: 10, property_tax: 0, equipment: 0, insurance: 0, accountant: 0 },
    turnover: { interval_years: 1, restoration: 0, move_out: 0, key: 0, utilities_month: 0, ad_months: 1, agent_months: 0, free_rent_months: 0, vacancy_months: 0 },
    risk: { rate_rise_pct: 0.5, rate_rise_years: 2, rent_decline_pct: 10, repair_rise_pct: 20 } };
  const Y = routes.cashflow(q).yearly;
  assert.deepEqual(Y.rate.slice(0, 4), [1, 1.5, 2, 2]);  // 2年目まで上がって止まる
  close(Y.rent[1], 0.9e5 * 12); close(Y.rental_mgmt[1], 0.9e4 * 12); close(Y.turnover[1], 0.9e5); close(Y.repair[1], 1.2e4 * 12);
  // 2年目の返済額は1年目末の残高・残り19年・金利1.5%で計算し直す
  const bal1 = Y.balance[0], q2 = 0.015 / 12, pay2 = bal1 * q2 / (1 - Math.pow(1 + q2, -228));
  close(Y.principal[1] + Y.interest[1], pay2 * 12, 1e-6);
  close(Y.balance[19], 0, 1e-6);
  const g = routes.cashflow(q).grid[1][3];
  assert.ok(g.total < g.total_base);
  assert.throws(() => routes.cashflow({ ...q, risk: { ...q.risk, rent_decline_pct: 100 } }), /家賃の下落率/);
  const o = routes.cashflow({ ...q, risk: { rate_override_pct: 4 } }).yearly;  // 全期間4%: 入力の金利(1%)の代わりに使う
  assert.ok(o.rate.slice(0, 20).every(v => v === 4));
  close(o.principal[0] + o.interest[0], 2e7 * (0.04 / 12) / (1 - Math.pow(1 + 0.04 / 12, -240)) * 12, 1e-6);
});

test('cashflow: 税金(減価償却・損益通算・譲渡所得税)', () => {
  const q = { price: 2e7, rent_month: 1e5, rate: 1, years: 20, down_pct: 0, brokerage: false, sale_brokerage: false,
    purchase: { loan_fee_pct: 0, scrivener: 0, registration_pct: 0, stamp: 0, acquisition_tax_pct: 0, fire_insurance: 0, settlement: 0 },
    running: { management: 0, repair_reserve: 1e4, rental_mgmt_pct: 10, property_tax: 0, equipment: 0, insurance: 0, accountant: 0 },
    turnover: { interval_years: 1, restoration: 0, move_out: 0, key: 0, utilities_month: 0, ad_months: 1, agent_months: 0, free_rent_months: 0, vacancy_months: 0 } };
  const tax = { rate_pct: 30, building_ratio_pct: 50, building_age: 20, legal_life: 47, equipment_ratio_pct: 0 };
  const n = routes.cashflow(q), r = routes.cashflow({ ...q, tax });
  assert.equal(n.tax, null); assert.equal(n.grid[1][3].sale_tax, 0);
  assert.equal(r.tax.life, 31);  // 中古の耐用年数: (47−20) + 20×0.2 = 31
  const dep = 1e7 / 31, Y = r.yearly;
  close(Y.depreciation[0], dep); assert.equal(Y.depreciation[31], 0);
  const inc = 120e4 - 12e4 - 12e4 - 10e4 - Y.interest[0] - dep;  // 黒字なので利子は全額経費
  close(Y.taxable[0], inc); close(Y.income_tax[0], inc * 0.3); close(Y.cf[0], n.yearly.cf[0] - inc * 0.3);
  // 売却: 譲渡所得 = 売却価格 − (取得費 − 償却累計)。10年(長期)は20.315%、5年(短期)は39.63%
  close(r.grid[1][3].sale_tax, 10 * dep * 0.20315); close(r.grid[0][3].sale_tax, 5 * dep * 0.3963);
  assert.equal(r.grid[0][0].sale_tax, 0);  // 3割安なら譲渡損で税金なし
  // 赤字: 土地の取得に充てた借入金(借入2000万 − 建物1000万 = 半分)の利子は損益通算できない
  const low = routes.cashflow({ ...q, rent_month: 3e4, tax }).yearly;
  const raw = 36e4 - 12e4 - 3.6e4 - 3e4 - low.interest[0] - dep;
  assert.ok(raw < 0); close(low.taxable[0], Math.min(0, raw + low.interest[0] * 0.5));
  assert.ok(low.income_tax[0] < 0);  // 赤字分の税金が戻る
  assert.equal(routes.cashflow({ ...q, tax: { ...tax, building_age: 50 } }).tax.life, 9);  // 法定耐用年数超: 47×0.2
  assert.equal(routes.cashflow({ ...q, tax: { ...tax, legal_life: 22 } }).tax.life, 6);  // 木造(22年)で築20年: (22−20) + 20×0.2
  // 設備を3割に分ける: 設備(法定15年)は築20年なので 15×0.2 = 3年、躯体は31年
  const e = routes.cashflow({ ...q, tax: { ...tax, equipment_ratio_pct: 30 } });
  assert.equal(e.tax.equipment_life, 3);
  close(e.yearly.depreciation[0], 0.7e7 / 31 + 0.3e7 / 3); close(e.yearly.depreciation[3], 0.7e7 / 31);
  close(e.grid[1][3].sale_tax, (10 * 0.7e7 / 31 + 0.3e7) * 0.20315);  // 10年で設備は償却済み
});

test('housing-national: 登記件数と着工戸数は12か月合計', async () => {
  const r = await routes['housing-national']({});
  assert.deepEqual([r.sales.n12.total[10], r.sales.n12.total[11], r.starts.sum12.rental.at(-1)], [null, 120, 120]);
  await assert.rejects(routes['housing-national']({ starts_region: 'どこか' }), /地域がありません/);
});

test('loan: 金利と機構の金利を月で揃え、変動の店頭の目安は短プラ+1%', async () => {
  const r = await routes.loan();
  assert.deepEqual([r.month[0], r.month.at(-1)], ['2015-01', '2021-06']);
  const i = r.month.indexOf('2020-07');
  close(r.series.float_store[i], 2.5); close(r.series.chintai_35[i], 2.0);
  assert.equal(r.series.prime_short.at(-1), null);  // 金利の系列は2020年まで
  assert.deepEqual(r.latest.prime_short, { month: '2020-12', value: 1.5 });
  assert.deepEqual(r.latest.chintai_35, { month: '2021-06', value: 2.0 });
  assert.ok(r.spread.chintai_35.every(v => v == null));  // 合成データには10年国債がない
  assert.deepEqual(r.boj_loans.sum4.housing_new, [null, null, null, 10]);
});

test('repayment: 元利均等で完済、金利が上がらなければ固定と同じ、損益分岐の上昇幅', () => {
  const base = { principal: 3e7, years: 30, float_rate: 1.0, rise_years: 10, fixed_rate: 1.0 };
  const flat = routes.repayment({ ...base, rise_pct: 0 });
  close(flat.float.total, flat.fixed.total, 1e-9); close(flat.breakeven_rise, 0, 1e-9);
  const r = routes.repayment({ ...base, rise_pct: 0.1, fixed_rate: 2.0 });
  assert.deepEqual(r.float.rate.slice(0, 3).map(v => +v.toFixed(6)), [1, 1.1, 1.2]);
  close(r.float.rate[10], 2.0); close(r.float.rate.at(-1), 2.0);  // 10年で上昇が止まる
  assert.ok(r.float.payment[1] > r.float.payment[0] && r.float.balance[0] === 3e7);
  const be = routes.repayment({ ...base, rise_pct: r.breakeven_rise, fixed_rate: 2.0 });
  close(be.float.total, be.fixed.total, 1e-6);
  assert.equal(routes.repayment({ ...base, rise_pct: 0, fixed_rate: 25 }).breakeven_rise, null);  // 2%/年×10年(21%まで)でも固定25%に並ばない
  assert.throws(() => routes.repayment({ ...base, principal: 0 }), /借入額/);
});

test('population-area: 男女・年齢層の合計と割合、ピラミッド。欠けた年は null', async () => {
  const r = await routes['population-area']({ code: '13101' });
  assert.deepEqual(r.total, [null, 800, 400]);  // 2015年は区のデータなし
  assert.deepEqual([r.male, r.female], [[null, 400, 200], [null, 400, 200]]);
  assert.deepEqual([r.groups.child[1], r.groups.work[1], r.groups.elderly[1], r.groups.old75[1]], [200, 400, 200, 80]);
  close(r.share.elderly[1], 0.25);
  assert.deepEqual(r.pyramid.m[2], [50, 100, 30, 20]);
  // 10歳階級: 年齢(下限) 0/15/65/75 → 0〜9歳, 10〜19歳, 60〜69歳, 70〜79歳
  assert.deepEqual(r.decades.labels, ['0〜9歳', '10〜19歳', '60〜69歳', '70〜79歳']);
  assert.deepEqual(r.decades.m.map(x => x[2]), [50, 100, 30, 20]);
  assert.deepEqual(r.decades.all.map(x => x[1]), [200, 400, 120, 80]);
  assert.equal(r.decades.all[0][0], null);
  await assert.rejects(routes['population-area']({ code: '99999' }), /データがありません/);
});

test('population-compare: 基準年=100の指数と増減率、総人口に占める割合', async () => {
  const r = await routes['population-compare']({ sex: 'f', group: 'elderly' });
  assert.equal(r.label, '女・65歳以上');
  assert.deepEqual(r.rows.map(x => x.code), ['00000', '13100']);
  const x = r.rows[1];
  assert.deepEqual([x.values, x.index], [[1000, 1000, 1000], [100, 100, 100]]);
  close(x.share_last, 1000 / 8000);
  await assert.rejects(routes['population-compare']({ base: 2000 }), /基準年/);
  await assert.rejects(routes['population-compare']({ group: 'x' }), /年齢層/);
});

test('population-map: 増減率と割合。全国の値も返す', async () => {
  const c = await routes['population-map']({ from: 2020, to: 2050 });
  close(c.values['13101'], -0.5); close(c.nation, 0); close(c.values['13000'], 0);
  assert.equal((await routes['population-map']({ from: 2015, to: 2020 })).values['13101'], null);
  const s = await routes['population-map']({ metric: 'share', group: 'old75', year: 2050 });
  close(s.values['13101'], 0.1); close(s.nation, 0.1);
  await assert.rejects(routes['population-map']({ metric: 'share' }), /年齢層を選んで/);
  await assert.rejects(routes['population-map']({ from: 2050, to: 2020 }), /年の指定/);
  assert.deepEqual(Object.keys(await routes['population-geo']()), ['13101']);
});

test('long: 基準年=100 にそろえ、実質は物価で割る。基準年に値のない価格の系列は地価に合わせる', async () => {
  const r = await routes.long({ city: 'X市', base: 2016 });
  const i = y => r.years.indexOf(y);
  close(r.series.land[i(2016)], 100); close(r.series.land[i(2018)], 400); assert.equal(r.series.land[i(2014)], null);
  close(r.series.rent[i(2021)], 170 / 120 * 100);
  close(r.series.jrei_tokyo[i(2017)], 200);  // 基準年に値があればそのまま
  close(r.ratio[i(2017)], (2 / 130) / (1 / 120) * 100);
  assert.deepEqual(r.summary.land, { year: 2021, value: 32 });
  assert.equal(r.summary.land_peak.year, 2021);
  close(r.land_yen[i(2020)], 100000); close(r.rent_yen[i(2016)], 60000 * 120 / 160);
  const real = await routes.long({ city: 'X市', base: 2016, real: true });
  close(real.series.land[i(2018)], 400 / 1.05 ** 2); assert.equal(real.series.land[i(2021)], null);  // 2021年は物価がない
  assert.equal(real.series.cpi, null);
  // 東京区部は2016年から: 2015年基準では2016年を地価に合わせる
  const early = await routes.long({ city: 'X市', base: 2015 });
  assert.equal(early.anchored.jrei_tokyo, 2016); close(early.series.jrei_tokyo[i(2016)], 200); close(early.series.jrei_tokyo[i(2017)], 400);
  // マンションの価格指数(rates.json、全国、2015〜2020年の月次の年平均)
  close(r.series.mansion[i(2017)], (100 + (24 + 35) / 2) / (100 + (12 + 23) / 2) * 100);
  const cmp = await routes.long({ city: 'X市', base: 2016, metric: 'rent' });
  assert.deepEqual(cmp.rows.map(w => w.city), ['X市', 'Y市']);
  assert.equal(cmp.rows[1].line, null); assert.equal(cmp.rows[1].rent, null);
  await assert.rejects(routes.long({ city: 'Z市' }), /都市が不正/);
  await assert.rejects(routes.long({ city: 'X市', base: 1900 }), /基準年が不正/);
});

test('long-rates: 変動の店頭の目安は短プラ+1%、物価上昇率は前年比', async () => {
  const r = await routes['long-rates']();
  close(r.values.float_store[0], 2.5);
  assert.deepEqual(r.inflation.years.slice(0, 2), [2015, 2016]);
  close(r.inflation.values[0], 5); assert.equal(r.inflation.values.at(-1), null);
});
