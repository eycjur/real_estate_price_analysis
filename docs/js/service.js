// 画面から呼ぶ分析処理 (モデル推定のキャッシュ、断面プロファイル、将来予測、利回り、金利と価格指数、相場、市況、ローン、人口、長期推移)。
// Web Worker (worker.js) の中で動く。

import { loadAll, loadBoundaries, loadDataset, loadLoan, loadLong, loadMarket, loadMeta, loadPopulation, loadRates } from './data.js';
import { FE_PREFIX, ageEffect, coef, fit, predict, predictRows, table } from './model.js';
import { interp, linspace, median, quantile, rng, sample } from './stats.js';

const MAX_POINTS = 2500;          // 散布図に返す実取引の上限
const PRED_ACTUAL_POINTS = 3000;  // 予測と実績の散布図に返す取引数(無作為抽出)
const fits = new Map(), holdouts = new Map();

const yen = (ds, r) => ds.cols.price[r] * 10000;
const year = (ds, r) => 2000 + ds.cols.year[r];

async function context(p) {
  const meta = await loadMeta();
  if (!meta.kinds[p.kind]) throw new Error(`種別が不正です: ${p.kind}`);
  const pooled = p.city === meta.all_label;
  const ds = await (pooled ? loadAll(p.kind) : loadDataset(p.kind, p.city));
  const from = p.year_from ?? 2010, to = p.year_to ?? 2100, rows = [];
  for (let r = 0; r < ds.n; r++) if (year(ds, r) >= from && year(ds, r) <= to) rows.push(r);
  return { meta, ds, rows: Int32Array.from(rows), pooled };
}

function spec(meta, kind, pooled) {
  const ref = meta.reference[kind];
  return { pooled, lambda: meta.district_lambda, vars: meta.variables[kind], baseLevels: ref.base_levels,
    ref: { age: ref.ref_age, area: ref.ref_area, areaStep: ref.area_step ?? null, land: ref.ref_land ?? 1, station: ref.ref_station,
      stationStep: ref.station_step ?? null } };
}

/** 全変数+地区効果(L2)の推定。条件ごとにキャッシュする。 */
async function getFit(p) {
  const key = JSON.stringify([p.city, p.kind, p.year_from ?? 2010, p.year_to ?? 2100]);
  if (!fits.has(key)) {
    const { meta, ds, rows, pooled } = await context(p);
    if (rows.length < 200) throw new Error(`該当する取引が少なすぎます (${rows.length}件)。条件を緩めてください。`);
    fits.set(key, fit(ds, rows, spec(meta, p.kind, pooled), meta, { cluster: pooled }));
  }
  return fits.get(key);
}

/** 取引の2割を除いて推定し、除いた2割での予測誤差を測る(過学習していないかの確認用)。 */
async function holdout(p) {
  const key = JSON.stringify([p.city, p.kind, p.year_from ?? 2010, p.year_to ?? 2100]);
  if (!holdouts.has(key)) {
    const { meta, ds, rows, pooled } = await context(p);
    if (pooled) return null;  // 全都市プールは推定に時間がかかるため省略
    const r = rng(0), train = [], test = [];
    for (const row of rows) (r() < 0.2 ? test : train).push(row);
    const f = fit(ds, Int32Array.from(train), spec(meta, p.kind, false), meta), mu = predictRows(f, test);
    let sse = 0, within = 0;
    test.forEach((row, i) => { const e = ds.lnPrice[row] - mu[i]; sse += e * e; if (Math.abs(Math.expm1(e)) <= 0.2) within++; });
    const rmse = Math.sqrt(sse / test.length);
    holdouts.set(key, { n_test: test.length, rmse, typical_error_pct: Math.expm1(rmse), within_20pct: within / test.length });
  }
  return holdouts.get(key);
}

export async function meta() {
  const m = await loadMeta();
  const infos = {};
  for (const kind of Object.keys(m.kinds)) for (const [city, e] of Object.entries(m.datasets[kind])) (infos[city] ??= { name: city, n: {}, wards: {} }, infos[city].n[kind] = e.n, infos[city].wards[kind] = e.wards);
  return { cities: m.cities.map(c => infos[c]), kinds: m.kinds, all_label: m.all_label, year_min: m.year_min, year_max: m.year_max, n_total: m.n_total,
    reference: m.reference };  // 基準の物件(土地面積を使う種別の判定と、種別を切り替えたときの面積の初期値)
}

export async function fitSummary(p) {
  const f = await getFit(p), { meta: m, ds, rows, pooled } = await context(p), resid = f.resid, n = rows.length;
  const hist = new Array(41).fill(0), edges = linspace(-1.5, 1.5, 42);
  let within20 = 0, within1sd = 0;
  for (const e of resid) {
    if (e >= -1.5 && e <= 1.5) hist[Math.min(40, Math.floor((e + 1.5) / 3 * 41))]++;
    if (Math.abs(Math.expm1(e)) <= 0.2) within20++;
    if (Math.abs(e) <= f.rmse) within1sd++;
  }
  const rowsTable = table(f);
  const years = rowsTable.filter(r => r.name.startsWith('year=')).map(r => ({ year: +r.name.slice(5), ...r })).sort((a, b) => a.year - b.year);
  const baseYear = pooled ? null : +f.design.vars.find(v => v.name === 'year').levels[0];
  let wards = rowsTable.filter(r => r.name.startsWith('ward='));
  let baseWard = f.design.baseWard;
  if (pooled) {
    // プール時は全区にダミーがある(定数項なし)ため、取引最多の区との差に直して表示する
    baseWard = f.design.vars.find(v => v.name === 'ward').levels[0];
    const ref = coef(f, `ward=${baseWard}`).coef;
    wards = wards.filter(r => r.name !== `ward=${baseWard}`).map(r => ({ ...r, coef: r.coef - ref }));
  }
  // 基準物件(切片に対応): 連続変数は基準値、カテゴリ項目・区・取引年は基準の水準、地区は区の平均
  const s = f.design.spec, age = f.design.vars[0];
  let maxYear = 0;
  if (pooled) { const w = ds.wards.indexOf(baseWard); for (const r of rows) if (ds.cols.ward[r] === w) maxYear = Math.max(maxYear, year(ds, r)); }
  // 面積・駅徒歩を区分で入れる種別は、基準の区分の下限を基準の値にする
  const baseOf = name => +f.design.vars.find(v => v.name === name).levels[0];
  const base = { age: +age.levels[0], area: s.ref.areaStep ? baseOf('area') : s.ref.area, station_min: s.ref.stationStep ? baseOf('station') : s.ref.station, ward: baseWard, year: baseYear ?? maxYear,
    price: Math.exp(pooled ? coef(f, `ward=${baseWard}`).coef : coef(f, 'const').coef) };
  if (f.design.house) base.land_area = s.ref.land;

  const shown = sample(rows.map((_, i) => i), PRED_ACTUAL_POINTS, 0);
  const areas = Array.from(rows, r => ds.cols.area[r]);
  const cats = {};
  for (const v of f.design.vars) if (m.variables[p.kind].includes(v.name)) cats[v.name] = { label: m.labels[v.name], levels: v.levels };
  return {
    params: { city: p.city, kind: p.kind, year_from: p.year_from ?? 2010, year_to: p.year_to ?? 2100 },
    n, r2: f.r2, adj_r2: f.adjR2, rmse: f.rmse, mae: f.mae, typical_error_pct: Math.expm1(f.rmse), se_type: f.seType,
    within_20pct: within20 / n, within_1sd: within1sd / n, base,
    pred_actual: { pred: shown.map(i => Math.round(Math.exp(ds.lnPrice[rows[i]] - resid[i]) / 1000) * 1000), actual: shown.map(i => yen(ds, rows[i])) },
    coefficients: rowsTable.filter(r => !FE_PREFIX.some(x => r.name.startsWith(x))), base_age: +age.levels[0], area_step: s.ref.areaStep, station_step: s.ref.stationStep,
    area_range: [Math.max(15, quantile(areas, 0.01)), quantile(areas, 0.99)],  // 面積のグラフの範囲(断面グラフと同じ)
    categoricals: cats, dropped: f.design.dropped, holdout: await holdout(p),
    year_effects: years, base_year: baseYear, ward_effects: wards.sort((a, b) => b.coef - a.coef), base_ward: baseWard,
    age_effects: [[0, 10], [10, 20], [20, 30], [30, 40]].map(([a, b]) => ({ from: a, to: b, pct: Math.expm1(ageEffect(f, a, b).effect) })),
    residual_hist: { counts: hist, edges },
  };
}

/** 1物件の条件(未指定の項目はデータの代表値)。名前で持つ view と、回帰に渡す番号の prop を返す。 */
function property(f, m, input, overrides = {}) {
  const { ds, rows } = f, kind = ds.kind, prop = { ...input, ...overrides };
  if (!f.byWard) {  // 区ごとの行と最新の取引年は、同じ推定に対して何度も使うので一度だけ作る
    f.byWard = new Map(); f.maxYear = 0;
    for (const r of rows) { (f.byWard.get(ds.cols.ward[r]) ?? f.byWard.set(ds.cols.ward[r], []).get(ds.cols.ward[r])).push(r); f.maxYear = Math.max(f.maxYear, year(ds, r)); }
  }
  const ward = prop.ward ? ds.wards.indexOf(prop.ward) : [...f.byWard].sort((a, b) => b[1].length - a[1].length)[0][0];
  const inWard = f.byWard.get(ward);
  if (!inWard) throw new Error(`区 '${prop.ward}' の取引がありません。`);
  const view = { ward: ds.wards[ward], city: ds.cities[ds.wardCity[ward]],
    year: Math.round(prop.year || f.maxYear), age: +(prop.age ?? 20),
    area: +(prop.area || median(Array.from(rows, r => ds.cols.area[r]))), station_min: +(prop.station_min ?? 8) };
  if (ds.cols.land_area) view.land_area = +(prop.land_area || median(Array.from(rows, r => ds.cols.land_area[r])));
  const district = prop.district ? ds.districts.findIndex(d => d.ward === ward && d.name === prop.district) : -1;
  if (district >= 0) view.district = prop.district;
  // 未指定のカテゴリ変数は、同じ区で面積が近い取引の最頻値にする(例: 25㎡なら間取りは 1K)
  const similar = inWard.filter(r => { const q = ds.cols.area[r] / view.area; return q >= 0.8 && q <= 1.25; });
  const cats = {};
  for (const v of m.variables[kind]) {
    const levels = m.levels[kind][v];
    let code = levels.indexOf(prop[v]);
    if (code < 0) {
      const count = new Map();
      for (const r of similar.length ? similar : inWard) count.set(ds.cols[v][r], (count.get(ds.cols[v][r]) ?? 0) + 1);
      code = [...count].sort((a, b) => b[1] - a[1] || levels[a[0]].localeCompare(levels[b[0]]))[0][0];
    }
    cats[v] = code; view[v] = levels[code];
  }
  return { view, prop: { ...view, ward, city: ds.wardCity[ward], district, cats }, inWard };
}

const VARY = {
  age: ['築年数(年)', () => Array.from({ length: 61 }, (_, i) => i)],
  // 面積を区分で入れる種別は、各区分の中央の値(例: 20〜25㎡ → 22.5㎡)を並べる
  area: ['建物面積(㎡)', (a, step) => {
    const lo = Math.max(15, quantile(a.area, 0.01)), hi = quantile(a.area, 0.99);
    if (!step) return linspace(lo, hi, 50);
    const out = [];
    for (let b = Math.floor(lo / step) * step; b <= hi; b += step) out.push(b + step / 2);
    return out;
  }],
  land_area: ['土地面積(㎡)', a => linspace(quantile(a.land_area, 0.01), quantile(a.land_area, 0.99), 50)],
  station_min: ['最寄駅徒歩(分)', () => Array.from({ length: 31 }, (_, i) => i)],
  year: ['取引年', a => [...new Set(a.year)].sort((x, y) => x - y)],
};

/** 指定した断面(他の条件を固定して1変数だけ動かす)での推定価格曲線と、近い条件の実取引。 */
export async function profile({ fit: p, vary, property: input }) {
  if (!VARY[vary]) throw new Error(`vary が不正です: ${vary}`);
  const f = await getFit(p), m = await loadMeta(), { ds, rows } = f;
  const column = name => Array.from(rows, r => name === 'year' ? year(ds, r) : ds.cols[name][r]);
  const [label, gridOf] = VARY[vary];
  const grid = gridOf({ get area() { return column('area'); }, get land_area() { return column('land_area'); }, get year() { return column('year'); } }, f.design.spec.ref.areaStep);
  const { view, prop, inWard } = property(f, m, input);
  const out = { price: [], ci_low: [], ci_high: [], pi_low: [], pi_high: [] };
  for (const x of grid) {
    const { mu, se } = predict(f, { ...prop, [vary]: x }), pse = Math.hypot(se, f.rmse);
    out.price.push(Math.exp(mu)); out.ci_low.push(Math.exp(mu - 1.96 * se)); out.ci_high.push(Math.exp(mu + 1.96 * se));
    out.pi_low.push(Math.exp(mu - 1.96 * pse)); out.pi_high.push(Math.exp(mu + 1.96 * pse));
  }
  // 断面に近い実取引: 同じ区で、動かす変数以外が固定値の近傍にあるもの
  const c = ds.cols, ratio = (a, b) => a / b >= 0.8 && a / b <= 1.25;
  let near = inWard.filter(r => (vary === 'age' || Math.abs(c.age[r] - view.age) <= 5) && (vary === 'area' || ratio(c.area[r], view.area))
    && (vary === 'land_area' || !c.land_area || ratio(c.land_area[r], view.land_area))
    && (vary === 'station_min' || Math.abs(c.station_min[r] - view.station_min) <= 5) && (vary === 'year' || Math.abs(year(ds, r) - view.year) <= 2));
  near = sample(near, MAX_POINTS, 0);
  const fixed = { ...view }; delete fixed[vary];
  return { vary, label, x: grid, ...out, fixed, district_effect: prop.district >= 0 ? f.gamma[prop.district] : 0,
    points: { x: near.map(r => vary === 'year' ? year(ds, r) : c[vary][r]), price: near.map(r => yen(ds, r)) }, n_points: near.length };
}

/** 物件の将来価格 = 現在の推定価格 × 経年減価 × 人口効果。市場全体の時間トレンドは横ばいと仮定。 */
export async function forecast({ fit: p, property: input, horizon_year = 2050 }) {
  const f = await getFit(p), m = await loadMeta(), { view, prop } = property(f, m, input);
  const el = m.population_elasticity[p.kind], pw = m.population[view.ward], mu0 = predict(f, prop).mu;
  const last = Math.min(horizon_year, pw.year[pw.year.length - 1]), years = [];
  for (let y = view.year; y <= last; y++) years.push(y);
  const pops = years.map(y => interp(y, pw.year, pw.population));
  const out = { age_only: [], pop_only: [], combined: [], combined_low: [], combined_high: [] };
  years.forEach((y, i) => {
    const a = ageEffect(f, view.age, view.age + (y - view.year)), lnPop = Math.log(pops[i] / pops[0]);
    const pop = el.coef * lnPop, total = a.effect + pop, se = Math.hypot(a.se, el.se * Math.abs(lnPop));
    out.age_only.push(Math.exp(mu0 + a.effect)); out.pop_only.push(Math.exp(mu0 + pop)); out.combined.push(Math.exp(mu0 + total));
    out.combined_low.push(Math.exp(mu0 + total - 1.96 * se)); out.combined_high.push(Math.exp(mu0 + total + 1.96 * se));
  });
  return { years, base_year: view.year, base_price: Math.exp(mu0), population: pops, ward: view.ward, ...out, elasticity: el };
}

/** 想定家賃[円/月] = 区×面積区分の平均家賃 × 築年補正(都市別・建築時期別の家賃 ÷ 都市平均)。 */
function estimatedRent(m, ward, city, area, age) {
  const a = Math.round(area), r = m.rent.find(x => x.ward === ward && x.area_min <= a && a <= x.area_max);
  if (!r) throw new Error(`${ward} の ${a}㎡ に対応する家賃統計がありません。家賃を直接指定してください。`);
  const ra = m.rent_age[city];
  return { rent: r.rent * (ra ? interp(age, ra.age, ra.factor) : 1),
    info: { ward_rent: r.rent, bracket: `${r.area_min}〜${r.area_max}㎡`, n_units: r.n_units } };
}

const COSTS = { vacancy_pct: 5, management_fee_pct: 5, building_fee_month: 12000, property_tax_year: 50000, purchase_cost_pct: 7, rent_month: null };
// 既定は投資用の変動型ローン: 短期プライムレート +1.5% (優遇後の実勢の目安。画面の「ローンの用途」の初期値と同じ)。借入額=推定価格
const LOAN = { rate: 'prime_short', spread_pct: 1.5, years: 35 };
/** 元利均等返済の月額。金利0のときは元金の均等割。 */
function monthlyPayment(principal, ratePct, years) {
  const n = years * 12, r = ratePct / 100 / 12;
  return r === 0 ? principal / n : principal * r / (1 - Math.pow(1 + r, -n));
}
/** 金利系列の年平均(値のある月だけ)に上乗せ幅を足したもの。{年: %} */
function yearlyRates(r, l) {
  const sum = {}, cnt = {};
  r.month.forEach((mo, i) => { const v = r.rates[l.rate][i]; if (v != null) { const y = +mo.slice(0, 4); sum[y] = (sum[y] ?? 0) + v; cnt[y] = (cnt[y] ?? 0) + 1; } });
  return Object.fromEntries(Object.keys(sum).map(y => [y, sum[y] / cnt[y] + l.spread_pct]));
}
/** 表面・実質(NOI)・返済後の利回り。返済後 = (NOI − 年間返済額) ÷ 総投資額。金利がない年は返済後を null にする。 */
function yields(price, rent, c, l, ratePct) {
  const income = rent * 12 * (1 - c.vacancy_pct / 100);
  const noi = income * (1 - c.management_fee_pct / 100) - c.building_fee_month * 12 - c.property_tax_year;
  const cost = price * (1 + c.purchase_cost_pct / 100), payment = ratePct == null ? null : monthlyPayment(price, ratePct, l.years);
  return { gross: rent * 12 / price, net: noi / cost, noi, payment, cash: payment == null ? null : (noi - payment * 12) / cost };
}

/** 回帰モデルの推定価格と統計ベースの想定家賃から、表面・実質(NOI)・ローン返済後の利回りを出す。
 *  断面は2つ: 築年数(選択中の取引年で固定、価格と家賃が築年で変わる)と、購入した年(築年数と家賃を固定し、価格と金利がその年の水準)。 */
export async function investmentYield({ fit: p, property: input, costs = {}, loan = {} }) {
  if (p.kind !== 'mansion') throw new Error('利回りの試算は中古マンションのみ対応しています。');
  const unknown = Object.keys(costs).filter(k => !(k in COSTS));
  if (unknown.length) throw new Error(`利回りの前提が不正です: ${unknown}`);
  const l = { ...LOAN, ...loan }, r = await loadRates();
  if (!(l.rate in r.rates)) throw new Error(`金利の系列が不正です: ${l.rate}`);
  if (!(l.years >= 1)) throw new Error('返済期間は1年以上にしてください。');
  const c = { ...COSTS, ...costs }, f = await getFit(p), m = await loadMeta(), { view, prop } = property(f, m, input);
  const rentAt = (ward, age) => c.rent_month ? { rent: +c.rent_month, info: null } : estimatedRent(m, ward, view.city, view.area, age);
  const rateBy = yearlyRates(r, l), rate = rateBy[view.year] ?? null;  // 選択中の取引年の金利
  const curve = { age: [], price: [], rent: [], payment: [], gross: [], net: [], cash: [] };
  let at = null, info = null;
  for (let age = 0; age <= 50; age++) {
    const price = Math.exp(predict(f, { ...prop, age }).mu), rr = rentAt(view.ward, age), y = yields(price, rr.rent, c, l, rate);
    curve.age.push(age); curve.price.push(price); curve.rent.push(rr.rent); curve.payment.push(y.payment);
    curve.gross.push(y.gross); curve.net.push(y.net); curve.cash.push(y.cash);
    if (age === Math.max(0, Math.min(50, Math.round(view.age)))) { at = { price, rent: rr.rent, ...y }; info = rr.info; }
  }
  // 購入した年の断面: 取引データにある各年。家賃は選択中の築年数のものを全期間に使う
  const years = [...new Set(Array.from(f.rows, row => year(f.ds, row)))].sort((a, b) => a - b);
  const byYear = { year: years, price: [], rate: [], payment: [], gross: [], net: [], cash: [] };
  for (const yr of years) {
    const price = Math.exp(predict(f, { ...prop, year: yr }).mu), ry = rateBy[yr] ?? null, y = yields(price, at.rent, c, l, ry);
    byYear.price.push(price); byYear.rate.push(ry); byYear.payment.push(y.payment); byYear.gross.push(y.gross); byYear.net.push(y.net); byYear.cash.push(y.cash);
  }
  const wards = [];
  for (const v of f.design.vars.find(x => x.name === 'ward').levels) {
    let rr;
    try { rr = rentAt(v, view.age); } catch { continue; }  // その区・面積区分の家賃統計がない
    const price = Math.exp(predict(f, property(f, m, input, { ward: v, district: null }).prop).mu);
    wards.push({ ward: v, price, rent: rr.rent, ...yields(price, rr.rent, c, l, rate) });
  }
  return { ward: view.ward, area: view.area, age: view.age, year: view.year, ...at, rate, rent_info: info, rent_overridden: !!c.rent_month, costs: c,
    loan: l, rate_label: r.labels[l.rate], curve, by_year: byYear, wards: wards.sort((a, b) => b.net - a.net) };
}

/** 金利と不動産価格指数の月次系列に、選択中の物件の各年の推定価格を指数(2010年=100)にして添えて返す。 */
export async function rates({ fit: p, property: input }) {
  const r = await loadRates(), f = await getFit(p), m = await loadMeta(), { view, prop } = property(f, m, input);
  const years = [...new Set(Array.from(f.rows, row => year(f.ds, row)))].sort((a, b) => a - b);
  const price = years.map(y => Math.exp(predict(f, { ...prop, year: y }).mu));
  const base = years.includes(2010) ? years.indexOf(2010) : 0;  // 不動産価格指数と同じ 2010年=100
  const region = r.rpi_region[view.city] ?? r.rpi_region[p.city];
  return { years, price, index: price.map(v => v / price[base] * 100), index_base_year: years[base],
    ward: view.ward, city: view.city, age: view.age, area: view.area,
    region, month: r.month, rates: r.rates, labels: r.labels, rpi: r.rpi[region] };
}

/** 区内の地区一覧(取引件数と地区効果)。 */
export async function districts({ fit: p, ward }) {
  const f = await getFit(p), { ds, rows } = f, w = ds.wards.indexOf(ward), n = new Map();
  for (const r of rows) if (ds.cols.ward[r] === w) n.set(ds.cols.district[r], (n.get(ds.cols.district[r]) ?? 0) + 1);
  return [...n].sort((a, b) => b[1] - a[1]).map(([q, c]) => ({ district: ds.districts[q].name, n: c, effect: f.gamma[q] }));
}

/** 地区ごとの立地効果(区効果+地区効果)を代表点つきで返す。値は基準の区の平均に対する ln価格差。 */
export async function districtMap(p) {
  const f = await getFit(p), { ds, rows, design } = f, wardVar = design.vars.find(v => v.name === 'ward');
  const baseWard = wardVar.levels[0], ref = design.spec.pooled ? coef(f, `ward=${baseWard}`).coef : 0;
  const n = new Map();
  for (const r of rows) n.set(ds.cols.district[r], (n.get(ds.cols.district[r]) ?? 0) + 1);
  const out = { base_ward: baseWard, n_missing: 0, ward: [], district: [], lat: [], lon: [], effect: [], district_effect: [], n: [] };
  for (const [q, count] of n) {
    const d = ds.districts[q];
    if (d.lat == null) { out.n_missing++; continue; }
    const w = coef(f, `ward=${ds.wards[d.ward]}`);
    out.ward.push(ds.wards[d.ward]); out.district.push(d.name); out.lat.push(d.lat); out.lon.push(d.lon);
    out.effect.push((w ? w.coef : 0) - ref + f.gamma[q]); out.district_effect.push(f.gamma[q]); out.n.push(count);
  }
  return out;
}

export async function populationSeries({ city }) {
  const m = await loadMeta(), wards = {};
  for (const kind of Object.keys(m.kinds)) for (const [c, e] of Object.entries(m.datasets[kind])) if (city === m.all_label || c === city) for (const w of e.wards) if (m.population[w]) wards[w] = m.population[w];
  return { wards, last_actual_year: m.population_last_actual_year };
}

/** 標準物件の推定価格・想定家賃・表面利回り。区ごとに計算し、条件に合う取引件数で加重平均する。 */
async function standardYield(m, city, wardCounts, std) {
  const f = await getFit({ city, kind: 'mansion' });
  let wsum = 0, price = 0, rent = 0, gross = 0;
  for (const [ward, n] of wardCounts) {
    let r;
    try { r = estimatedRent(m, ward, city, std.area, std.age); } catch { continue; }  // その区・面積区分の家賃統計がない
    const pr = Math.exp(predict(f, property(f, m, { ...std, ward }).prop).mu);
    wsum += n; price += n * pr; rent += n * r.rent; gross += n * r.rent * 12 / pr;
  }
  return wsum ? { std_price: price / wsum, std_rent: rent / wsum, std_gross: gross / wsum } : {};
}

/** 条件に合う実取引の相場(中央値・四分位)と、標準物件で揃えた家賃・表面利回りを地域別に返す。 */
export async function market(query) {
  const m = await loadMeta();
  const q = { region: m.all_label, kind: 'mansion', area_min: 15, area_max: 30, age_min: 0, age_max: 40, station_max: 15, year_from: 2023, year_to: 2100, ...query };
  const byCity = q.region === m.all_label, ds = await (byCity ? loadAll(q.kind) : loadDataset(q.kind, q.region)), c = ds.cols;
  const groups = new Map();  // 地域名 → { rows: 取引年の条件も満たす行, trend: 年 → ㎡単価 }
  for (let r = 0; r < ds.n; r++) {
    if (c.area[r] < q.area_min || c.area[r] > q.area_max || c.age[r] < q.age_min || c.age[r] > q.age_max || c.station_min[r] > q.station_max) continue;
    const name = byCity ? ds.cities[ds.cityOf[r]] : ds.wards[c.ward[r]], y = year(ds, r);
    const g = groups.get(name) ?? groups.set(name, { rows: [], trend: new Map() }).get(name);
    (g.trend.get(y) ?? g.trend.set(y, []).get(y)).push(yen(ds, r) / c.area[r]);  // 推移グラフ用: 取引年以外の条件で絞った全期間
    if (y >= q.year_from && y <= q.year_to) g.rows.push(r);
  }
  // 利回りは実取引の中央値と統計家賃を割らない(都市によって取引される物件の築年数がまるで違うため)。
  // 条件の中央の「標準物件」を決め、区ごとに 回帰の推定価格 と 同じ築年数の家賃 から計算して揃える
  const std = q.kind !== 'mansion' ? null : { age: Math.round((q.age_min + q.age_max) / 2), area: Math.round((q.area_min + q.area_max) / 2),
    station_min: Math.min(10, q.station_max), year: Math.min(q.year_to, m.year_max) };
  const rows = [];
  let total = 0;
  for (const [name, g] of groups) {
    if (!g.rows.length) continue;
    total += g.rows.length;
    const price = g.rows.map(r => yen(ds, r));
    const row = { name, n: g.rows.length, price: median(price), price_p25: quantile(price, 0.25), price_p75: quantile(price, 0.75),
      unit_price: median(g.rows.map(r => yen(ds, r) / c.area[r])), age: median(g.rows.map(r => c.age[r])), area: median(g.rows.map(r => c.area[r])),
      std_price: null, std_rent: null, std_gross: null, by_year: {}, n_by_year: {} };
    for (const [y, v] of g.trend) { row.by_year[y] = median(v); row.n_by_year[y] = v.length; }
    if (std) {
      const wardCounts = new Map();
      for (const r of g.rows) wardCounts.set(ds.wards[c.ward[r]], (wardCounts.get(ds.wards[c.ward[r]]) ?? 0) + 1);
      Object.assign(row, await standardYield(m, byCity ? name : q.region, wardCounts, std));
    }
    rows.push(row);
  }
  if (!rows.length) throw new Error('条件に合う取引がありません。条件を緩めてください。');
  return { by: byCity ? 'city' : 'ward', n: total, rows: rows.sort((a, b) => b.unit_price - a.unit_price), standard: std, query: q };
}

/** 直近 n 個の合計。足りない・欠損を含む位置は null。 */
function rollSum(a, n) {
  return a.map((_, i) => {
    if (i < n - 1) return null;
    let s = 0;
    for (let j = i - n + 1; j <= i; j++) { if (a[j] == null) return null; s += a[j]; }
    return s;
  });
}
const ratio = (a, b) => a.map((v, i) => v == null || b[i] == null || b[i] === 0 ? null : v / b[i]);
const sum = a => a.reduce((s, v) => s + v, 0);

/** 価格帯の表示名。「~1000」「~2000」…「10000~」(万円) →「1,000万円以下」「1,000万〜2,000万円」…「1億円超」 */
export function bandLabels(labels) {
  const fmt = v => v >= 10000 ? `${v / 10000}億` : `${v.toLocaleString('ja-JP')}万`;
  return labels.map((b, i) => {
    const v = +b.replace('~', '');
    if (b.endsWith('~')) return `${fmt(v)}円超`;
    return i === 0 ? `${fmt(v)}円以下` : `${fmt(+labels[i - 1].replace('~', ''))}〜${fmt(v)}円`;
  });
}

/** 件数の系列から、成約率(成約÷新規登録)と在庫月数(在庫÷月平均の成約)を出す。w = 1年に当たる期間の数(月次12・四半期4)。
 *  比率は季節変動をならすため1年分の合計で計算する。sold_avg / new_avg は1期間あたりの平均件数。 */
function counts(period, sold, nw, stock, w) {
  const months = 12 / w, soldY = rollSum(sold, w), newY = rollSum(nw, w);
  return {
    period, sold, new: nw, stock, sold_avg: soldY.map(v => v == null ? null : v / w), new_avg: newY.map(v => v == null ? null : v / w),
    contract_rate: ratio(sold, nw), contract_rate12: ratio(soldY, newY),
    months_of_stock: ratio(stock, sold.map(v => v == null ? null : v / months)), months_of_stock12: ratio(stock, soldY.map(v => v == null ? null : v / 12)),
  };
}

/** レインズ(首都圏)の市況。月次の件数と、成約率・在庫月数(12か月)、価格帯別、地域の比較。
 *  band を指定すると、件数・成約率・在庫月数をその価格帯の四半期の値にする(価格帯別は首都圏と都県だけ、四半期で公表)。
 *  価格の推移と地域の比較は価格帯によらない全体の値。 */
export async function housingMarket({ kind = 'mansion', region = '首都圏', band = null }) {
  const d = await loadMarket(), r = d.reins, s = r.series[kind]?.[region];
  if (!s) throw new Error(`市況のデータがありません: ${kind} ${region}`);
  const priceKey = 'unit_price' in s.sold ? 'unit_price' : 'price';  // 戸建は㎡単価がないので価格
  const bn = r.band_n[kind]?.[region], last = r.month.length - 1;
  if (band && !bn) throw new Error(`価格帯別の件数は首都圏と都県だけです: ${region}`);
  const bi = band ? r.bands[kind].indexOf(band) : -1;
  if (band && bi < 0) throw new Error(`価格帯が不正です: ${band}`);
  const c = band ? { freq: 'Q', ...counts(r.quarter, bn.sold[bi], bn.new[bi], bn.stock[bi], 4) } : { freq: 'M', ...counts(r.month, s.sold.n, s.new.n, s.stock.n, 12) };
  const w = c.freq === 'Q' ? 4 : 12, k = c.period.length - 1, yoy = a => a[k] != null && a[k - w] ? a[k] / a[k - w] - 1 : null;
  const pyoy = a => a[last] != null && a[last - 12] ? a[last] / a[last - 12] - 1 : null;
  const monthly = { ...c, month: r.month, price: { sold: s.sold[priceKey], new: s.new[priceKey], stock: s.stock[priceKey] }, age: { sold: s.sold.age, new: s.new.age, stock: s.stock.age } };
  const latest = {
    period: c.period[k], sold: c.sold[k], new: c.new[k], stock: c.stock[k], sold_yoy: yoy(c.sold), new_yoy: yoy(c.new), stock_yoy: yoy(c.stock),
    contract_rate12: c.contract_rate12[k], months_of_stock12: c.months_of_stock12[k],
    price_month: r.month[last], price: s.sold[priceKey][last], price_yoy: pyoy(s.sold[priceKey]),
    // 売り出し価格(新規登録)と成約価格の差。新規登録が高いほど、売り手の希望と買い手の評価が離れている
    ask_gap: s.new[priceKey][last] / s.sold[priceKey][last] - 1,
  };

  // 価格帯別(四半期): 直近4四半期の成約÷新規登録と、在庫÷月間成約。構成比の推移は成約件数で
  let bands = null;
  if (bn) {
    const q = r.quarter, n = q.length, recent = (a) => a.slice(n - 4), names = bandLabels(r.bands[kind]);
    const rows = r.bands[kind].map((label, i) => {
      const sd = sum(recent(bn.sold[i])), nd = sum(recent(bn.new[i])), st = bn.stock[i][n - 1];
      return { label, name: names[i], sold: sd, new: nd, stock: st, contract_rate: nd ? sd / nd : null, months_of_stock: sd ? st / (sd / 12) : null };
    });
    const total = q.map((_, j) => sum(bn.sold.map(b => b[j])));
    bands = { quarter: q, labels: r.bands[kind], names, recent: { from: q[n - 4], to: q[n - 1] }, rows,
      sold: bn.sold, new: bn.new, stock: bn.stock, share: bn.sold.map(b => b.map((v, j) => total[j] ? v / total[j] : null)) };
  }

  // 地域の比較: 直近12か月の合計と、その前の12か月からの価格の変化(件数で加重した平均)
  const wavg = (p, n, from, to) => { let a = 0, w = 0; for (let i = from; i <= to; i++) if (p[i] != null && n[i]) { a += p[i] * n[i]; w += n[i]; } return w ? a / w : null; };
  const regions = r.regions.filter(g => r.series[kind][g]).map(g => {
    const x = r.series[kind][g], sd = sum(x.sold.n.slice(last - 11)), nd = sum(x.new.n.slice(last - 11));
    const p1 = wavg(x.sold[priceKey], x.sold.n, last - 11, last), p0 = wavg(x.sold[priceKey], x.sold.n, last - 23, last - 12);
    return { region: g, note: r.region_notes[g], sold: sd, new: nd, stock: x.stock.n[last], contract_rate: sd / nd,
      months_of_stock: x.stock.n[last] / (sd / 12), price: p1, price_yoy: p0 ? p1 / p0 - 1 : null };
  });
  const regionsByKind = Object.fromEntries(Object.keys(d.kinds).map(k => [k, r.regions.filter(g => r.series[k]?.[g])]));
  const bandRegions = Object.keys(r.band_n[kind] ?? {});
  return { kinds: d.kinds, regions_by_kind: regionsByKind, band_regions: bandRegions, band, band_name: band ? bands.names[bi] : null, kind, kind_label: d.kinds[kind], region, region_note: r.region_notes[region], price_key: priceKey, monthly, latest, bands, regions };
}

/** 全国の断面: 既存住宅販売量指数(登記ベース、地域×戸建/マンション)と、新設住宅着工戸数(都道府県×利用関係、12か月合計)。 */
export async function housingNational({ sales_region = '全国', starts_region = '全国' }) {
  const d = await loadMarket(), si = d.sales_index.series[sales_region], st = d.starts.series[starts_region];
  if (!si) throw new Error(`既存住宅販売量指数に地域がありません: ${sales_region}`);
  if (!st) throw new Error(`着工統計に地域がありません: ${starts_region}`);
  const keys = ['total', ...Object.keys(d.starts.labels)];
  return {
    sales: { month: d.sales_index.month, labels: d.sales_index.labels, regions: d.sales_index.regions, region: sales_region,
      index: Object.fromEntries(Object.keys(d.sales_index.labels).map(k => [k, si[k]])),
      n12: Object.fromEntries(Object.keys(d.sales_index.labels).map(k => [k, rollSum(si[k + '_n'], 12)])) },
    starts: { month: d.starts.month, labels: d.starts.labels, regions: d.starts.regions, region: starts_region,
      monthly: Object.fromEntries(keys.map(k => [k, st[k]])), sum12: Object.fromEntries(keys.map(k => [k, rollSum(st[k], 12)])) },
  };
}

/** ローン・金利タブ。日銀・財務省の金利(rates.json)と機構の賃貸住宅融資の金利を同じ月の並びに揃え、10年国債との差も出す。
 *  変動の店頭金利は大手行の慣行(短期プライムレート+1%)による目安。 */
export async function loanOverview() {
  const [r, l] = await Promise.all([loadRates(), loadLoan()]);
  const month = [...new Set([...r.month, ...l.chintai.month])].sort();
  const at = (ms, vs) => { const m = new Map(ms.map((x, i) => [x, vs[i]])); return month.map(x => m.get(x) ?? null); };
  const rate = k => at(r.month, r.rates[k]);
  const prime = rate('prime_short');
  const series = {
    float_store: prime.map(v => v == null ? null : v + 1), prime_short: prime, prime_long: rate('prime_long'), jgb10: rate('jgb10'),
    lend_new_long: rate('lend_new_long'), policy_rate: rate('policy_rate'),
    chintai_35: at(l.chintai.month, l.chintai.rates.free_35), chintai_15: at(l.chintai.month, l.chintai.rates.free_15),
  };
  const jgb = series.jgb10, spread = {};
  for (const k of ['float_store', 'prime_long', 'lend_new_long', 'chintai_35']) spread[k] = series[k].map((v, i) => v == null || jgb[i] == null ? null : v - jgb[i]);
  const latest = Object.fromEntries(Object.entries(series).map(([k, v]) => { const i = v.findLastIndex(x => x != null); return [k, { month: month[i], value: v[i] }]; }));
  const share = l.rate_type_share, bq = l.boj_loans;
  return { month, series, spread, latest, products: l.products, rate_type_share: share,
    boj_loans: { quarter: bq.quarter, labels: bq.labels, sum4: Object.fromEntries(Object.entries(bq.values).map(([k, v]) => [k, rollSum(v, 4)])) } };
}

/** 元利均等返済の比較。変動は1年ごとに rise_pct ずつ rise_years 年間上がると仮定し、毎年その時点の残高・残期間で返済額を見直す
 *  (実際の商品の「5年ルール・125%ルール」は考えない)。固定は全期間同じ金利。
 *  breakeven_rise: 変動の総返済額が固定と同じになる年あたりの上昇幅(rise_years は同じ。0〜2%/年で見つからなければ null)。 */
export function repayment({ principal, years, float_rate, rise_pct = 0, rise_years = 10, fixed_rate }) {
  if (!(principal > 0 && years >= 1)) throw new Error('借入額と返済期間を正しく入れてください。');
  const n = Math.round(years * 12);
  const simulate = rise => {
    let bal = principal, pay = 0, total = 0;
    const yr = { year: [], rate: [], payment: [], balance: [] };
    for (let m = 0; m < n; m++) {
      if (m % 12 === 0) {
        const y = m / 12, rt = float_rate + rise * Math.min(y, rise_years);
        pay = monthlyPayment(bal, rt, (n - m) / 12);
        yr.year.push(y + 1); yr.rate.push(rt); yr.payment.push(pay); yr.balance.push(bal);
      }
      const rt = yr.rate.at(-1) / 100 / 12, interest = bal * rt;
      total += pay; bal -= pay - interest;
    }
    return { ...yr, total };
  };
  const fl = simulate(rise_pct), fx = simulate(0);
  const fixed = (() => { const p = monthlyPayment(principal, fixed_rate, years); return { rate: fixed_rate, payment: p, total: p * n }; })();
  // 総返済額は上昇幅について単調増加なので二分法
  const gap = rise => simulate(rise).total - fixed.total;
  let breakeven = null;
  if (gap(0) >= 0) breakeven = 0;
  else if (gap(2) > 0) { let lo = 0, hi = 2; for (let i = 0; i < 50; i++) { const mid = (lo + hi) / 2; gap(mid) > 0 ? hi = mid : lo = mid; } breakeven = (lo + hi) / 2; }
  return { float: fl, float_flat: { total: fx.total, payment: fx.payment[0] }, fixed, breakeven_rise: breakeven, principal, years, rise_pct, rise_years };
}

/** 返済額の計算。method: 'annuity'(元利均等: 毎月の返済額が一定) / 'principal'(元金均等: 毎月の元金が一定で、返済額は徐々に減る)。
 *  月々の返済額とその利息・元金の内訳、総返済額・利息の合計、年ごとの元金・利息・年末残高を返す。 */
export function loanCalc({ principal, rate, years, method = 'annuity' }) {
  if (!(principal > 0 && years >= 1 && rate >= 0)) throw new Error('借入額・金利・返済期間を正しく入れてください。');
  if (!['annuity', 'principal'].includes(method)) throw new Error(`返済方式が不正です: ${method}`);
  const n = Math.round(years * 12), r = rate / 100 / 12, fixedPay = monthlyPayment(principal, rate, n / 12);
  const yearly = { year: [], principal: [], interest: [], balance: [] };
  let bal = principal, totalInterest = 0, first = null, lastPay = 0;
  for (let m = 0; m < n; m++) {
    const interest = bal * r, pr = method === 'annuity' ? fixedPay - interest : principal / n;
    if (m === 0) first = { payment: pr + interest, interest, principal: pr };
    if (m % 12 === 0) { yearly.year.push(m / 12 + 1); yearly.principal.push(0); yearly.interest.push(0); yearly.balance.push(0); }
    bal -= pr; totalInterest += interest; lastPay = pr + interest;
    const y = yearly.year.length - 1;
    yearly.principal[y] += pr; yearly.interest[y] += interest; yearly.balance[y] = Math.max(bal, 0);
  }
  return { principal, rate, years, method, first, last_payment: lastPay, total_interest: totalInterest, total_payment: principal + totalInterest, yearly };
}

/** 仲介手数料の上限(税込)。400万円超の速算式 (価格 × 3% + 6万円) に消費税10%。 */
const brokerageFee = price => (price * 0.03 + 6e4) * 1.1;
/** 年ごとのキャッシュフロー(0年目=購入時)の内部収益率。−99%〜1000%で符号が変わらず求まらなければ null。 */
function irr(flows) {
  const npv = r => flows.reduce((s, v, t) => s + v / Math.pow(1 + r, t), 0);
  let lo = -0.99, hi = 10;
  if (npv(lo) * npv(hi) > 0) return null;
  for (let i = 0; i < 200; i++) { const mid = (lo + hi) / 2; npv(lo) * npv(mid) <= 0 ? hi = mid : lo = mid; }
  return (lo + hi) / 2;
}
/** 毎年金利を見直す元利均等返済。金利は年 rise ずつ riseYears 年目まで上がり、見直しのたびにその時点の残高・残り期間で返済額を計算し直す。
 *  年ごとの金利・元金・利息・年末残高を返す。 */
function floatingLoan(principal, rate, years, rise, riseYears) {
  const n = Math.round(years * 12), out = { rate: [], principal: [], interest: [], balance: [] };
  let bal = principal, pay = 0;
  for (let m = 0; m < n; m++) {
    if (m % 12 === 0) {
      const rt = rate + rise * Math.min(m / 12, riseYears);
      pay = monthlyPayment(bal, rt, (n - m) / 12);
      out.rate.push(rt); out.principal.push(0); out.interest.push(0); out.balance.push(0);
    }
    const y = out.rate.length - 1, interest = bal * out.rate[y] / 1200;
    out.interest[y] += interest; out.principal[y] += pay - interest; bal -= pay - interest; out.balance[y] = Math.max(bal, 0);
  }
  return out;
}
// 収支シミュレーションの内訳の項目(金額は円。_pct は%、_months は家賃の何か月分)
const SIM_KEYS = {
  purchase: ['loan_fee_pct', 'scrivener', 'registration_pct', 'stamp', 'acquisition_tax_pct', 'fire_insurance', 'settlement'],
  running: ['management', 'repair_reserve', 'rental_mgmt_pct', 'property_tax', 'equipment', 'insurance', 'earthquake_insurance', 'accountant'],
  turnover: ['interval_years', 'restoration', 'move_out', 'key', 'utilities_month', 'ad_months', 'agent_months', 'free_rent_months', 'vacancy_months'],
  risk: ['rate_rise_pct', 'rate_rise_years', 'rate_override_pct', 'repair_rise_pct', 'management_rise_pct'],
};
const NO_RISK = { rate_rise_pct: 0, rate_rise_years: 0, rate_override_pct: 0, repair_rise_pct: 0, management_rise_pct: 0 };  // rate_override_pct: 0より大きければ当初からその金利
// 売却価格の列: 入力した物件価格の年率(price_change_pct)にこの幅を足した年率で、保有年数のあいだ複利で変わる
const SIM_HOLD_YEARS = [5, 10, 15, 20, 30], SIM_PRICE_OFFSETS = [-2, -1, -0.5, 0, 0.5, 1];
const SIM_YEARS = 50;  // 年ごとの収支を出す年数(画面で選べる年の上限)
/** 1〜SIM_YEARS年目の年間の収入・支出(円)。家賃と入れ替えのコストは1戸あたりの額 × 戸数(units。区分は1)。家賃は毎年 rent_change_pct% 変わり、修繕積立金は毎年 repair_rise_pct%、管理費は毎年 management_rise_pct% 上がる。
 *  賃貸管理手数料と入れ替え時の家賃何か月分の費用は、その年の家賃で計算する。 */
function simYearly({ price, rent_month, units, rent_change_pct, rate, years, down_pct, running: r, turnover: t }, risk) {
  const loan = price * (1 - down_pct / 100);
  const L = loan > 0 ? floatingLoan(loan, risk.rate_override_pct || rate, years, risk.rate_rise_pct, risk.rate_rise_years) : null;
  const Y = { year: [], rate: [], rent: [], management: [], repair: [], rental_mgmt: [], other: [], turnover: [], interest: [], principal: [], balance: [], cf: [] };
  const other = r.property_tax + r.equipment + r.insurance + r.earthquake_insurance + r.accountant;
  for (let y = 1; y <= SIM_YEARS; y++) {
    const rent = rent_month * units * Math.pow(1 + rent_change_pct / 100, y - 1), i = y - 1, on = L && i < L.rate.length;
    const v = { year: y, rate: on ? L.rate[i] : null, rent: rent * 12, management: r.management * Math.pow(1 + risk.management_rise_pct / 100, y - 1) * 12, repair: r.repair_reserve * Math.pow(1 + risk.repair_rise_pct / 100, y - 1) * 12,
      rental_mgmt: rent * r.rental_mgmt_pct / 100 * 12, other: other * 12,
      turnover: (t.restoration * units + t.move_out * units + t.key * units + t.utilities_month * t.vacancy_months * units + rent * (t.ad_months + t.agent_months + t.free_rent_months + t.vacancy_months)) / t.interval_years,
      interest: on ? L.interest[i] : 0, principal: on ? L.principal[i] : 0, balance: on ? L.balance[i] : 0 };
    v.cf = v.rent - v.management - v.repair - v.rental_mgmt - v.other - v.turnover - v.interest - v.principal;
    for (const k of Object.keys(Y)) Y[k].push(v[k]);
  }
  return Y;
}
/** 中古の建物の耐用年数(簡便法)。法定耐用年数(legal)を過ぎていればその20%、それ以外は (法定 − 経過年数) + 経過年数 × 20%。1年未満は切り捨て、2年未満は2年。 */
const usedLife = (age, legal) => Math.max(2, Math.floor(age >= legal ? legal * 0.2 : legal - age + age * 0.2));
const EQUIPMENT_LIFE = 15;  // 建物附属設備(電気・給排水・ガス設備など)の法定耐用年数
const SALE_TAX_SHORT = 39.63, SALE_TAX_LONG = 20.315;  // 譲渡所得の税率(所得税・復興特別所得税・住民税)。保有5年以下 / 5年超
/** 年ごとの不動産所得に税率を掛けた所得税・住民税(赤字なら給与などとの損益通算で戻る分をマイナスで)を足し、cf を税引き後にする。
 *  経費: 家賃以外の支出のうちローン元金を除くもの + 建物の減価償却(定額法。躯体と設備を分け、それぞれの耐用年数で償却) + 1年目だけ購入時の諸費用(仲介手数料は取得費に入れる)。
 *  赤字のとき、土地の取得に充てた借入金の利子は損益通算できない(借入金はまず建物に充てたとみなす)。 */
function addIncomeTax(S, tax, { acquisition, expensed, loan }) {
  const building = acquisition * tax.building_ratio_pct / 100, equipment = building * tax.equipment_ratio_pct / 100;
  const life = usedLife(tax.building_age, tax.legal_life), equipmentLife = usedLife(tax.building_age, EQUIPMENT_LIFE);
  const landLoanShare = loan > 0 ? Math.max(0, loan - building) / loan : 0;
  for (const k of ['dep_body', 'dep_equipment', 'depreciation', 'interest_disallowed', 'taxable', 'income_tax']) S[k] = [];
  S.year.forEach((_, j) => {
    const db = j < life ? (building - equipment) / life : 0, de = j < equipmentLife ? equipment / equipmentLife : 0;
    const raw = S.rent[j] - S.management[j] - S.repair[j] - S.rental_mgmt[j] - S.other[j] - S.turnover[j] - S.interest[j] - db - de - (j === 0 ? expensed : 0);
    const inc = raw < 0 ? Math.min(0, raw + S.interest[j] * landLoanShare) : raw, tx = inc * tax.rate_pct / 100;
    S.dep_body.push(db); S.dep_equipment.push(de); S.depreciation.push(db + de); S.interest_disallowed.push(inc - raw);
    S.taxable.push(inc); S.income_tax.push(tx); S.cf[j] -= tx;
  });
  return { life, equipment_life: equipmentLife, building, equipment };
}
/** 区分マンション投資の収支。購入時の費用(頭金+諸費用)、月々のランニングコスト、入居者の入れ替え(空室など)を年平均にならしたコスト、
 *  ローン返済から、売却する年数 × 売却価格の年率(price_change_pct + SIM_PRICE_OFFSETS、%/年)ごとの総収支と、自己資金に対する年利回り(IRR)を出す。
 *  家賃は毎年 rent_change_pct% 変わる。risk(金利上昇・当初からの高い金利・修繕積立金と管理費の値上げ)を反映した値と、反映しない値(_base)の両方を返す。
 *  総収支 = −自己資金 + 保有中の収支の累計 + (売却価格 − 売却時の仲介手数料 − ローン残高 − 譲渡所得税)。
 *  tax(税率%・建物の割合%・建物のうち設備の割合%・築年数・構造の法定耐用年数)を渡すと、保有中の所得税・住民税(節税効果を含む)と売却時の譲渡所得税を入れる。null なら税金は含めない。 */
export function cashflowSim(q) {
  const { price, rent_month, units, rent_change_pct, price_change_pct, rate, years, down_pct, brokerage, sale_brokerage, purchase, running, turnover } = q, risk = { ...NO_RISK, ...q.risk };
  if (!(price > 0 && rent_month >= 0 && rate >= 0 && years >= 1 && down_pct >= 0 && down_pct <= 100)) throw new Error('物件価格・家賃・金利・返済期間・頭金を正しく入れてください。');
  const groups = { purchase, running, turnover, risk };
  for (const [g, keys] of Object.entries(SIM_KEYS)) {
    const bad = keys.filter(k => !(groups[g]?.[k] >= 0));
    if (bad.length) throw new Error(`費用の内訳・リスクの前提が不正です: ${bad}`);
  }
  if (!(Number.isInteger(units) && units >= 1)) throw new Error('戸数は1以上の整数にしてください。');
  if (!(turnover.interval_years > 0)) throw new Error('入れ替えの頻度は0より大きい年数にしてください。');
  if (!(rent_change_pct > -100)) throw new Error('家賃の変動率は−100%より大きくしてください。');
  if (!(price_change_pct + Math.min(...SIM_PRICE_OFFSETS) > -100)) throw new Error(`物件価格の変動率は${-100 - Math.min(...SIM_PRICE_OFFSETS)}%より大きくしてください。`);
  const tax = q.tax ?? null;
  if (tax && !(tax.rate_pct >= 0 && tax.rate_pct <= 100 && tax.building_ratio_pct >= 0 && tax.building_ratio_pct <= 100 && tax.building_age >= 0 && tax.legal_life >= 1 && tax.equipment_ratio_pct >= 0 && tax.equipment_ratio_pct <= 100))
    throw new Error('税率・建物と設備の割合・築年数・耐用年数を正しく入れてください。');

  const loan = price * (1 - down_pct / 100), rent = rent_month * units;
  const init = { down: price - loan, brokerage: brokerage ? brokerageFee(price) : 0, loan_fee_pct: loan * purchase.loan_fee_pct / 100, scrivener: purchase.scrivener,
    registration_pct: price * purchase.registration_pct / 100, stamp: purchase.stamp, acquisition_tax_pct: price * purchase.acquisition_tax_pct / 100,
    fire_insurance: purchase.fire_insurance, settlement: purchase.settlement };
  const initial = sum(Object.values(init));
  // 1年目の内訳(画面のカード)
  const run = { ...running, rental_mgmt_pct: rent * running.rental_mgmt_pct / 100 };
  const t = turnover;
  // 入れ替え1回(1戸)あたりの費用。年平均は全戸ぶん(× units)
  const tv = { restoration: t.restoration, move_out: t.move_out, key: t.key, utilities_month: t.utilities_month * t.vacancy_months,
    ad_months: rent_month * t.ad_months, agent_months: rent_month * t.agent_months, free_rent_months: rent_month * t.free_rent_months, vacancy_months: rent_month * t.vacancy_months };
  const perTurn = sum(Object.values(tv));

  const Y = simYearly(q, risk), B = simYearly(q, NO_RISK);
  const acquisition = price + init.brokerage;  // 取得費(仲介手数料を含む)
  let taxInfo = null;
  if (tax) {
    const ctx = { acquisition, expensed: initial - init.down - init.brokerage, loan };
    taxInfo = { ...tax, ...addIncomeTax(Y, tax, ctx), expensed: ctx.expensed };
    addIncomeTax(B, tax, ctx);
  }
  const cum = (a, h) => sum(a.slice(0, h));
  // h年後に年率 c% で変わった価格で売ったときの売却代金の手取り
  const saleNet = (S, h, c) => {
    const sale = price * Math.pow(1 + c / 100, h), saleCost = sale_brokerage ? brokerageFee(sale) : 0, bal = S.balance[h - 1];
    // 譲渡所得 = 売却価格 − 売却費用 − (取得費 − 減価償却の累計)。保有5年以下は短期の税率(売却した年の1月1日時点の所有期間の判定は考えない)
    const saleTax = tax ? Math.max(0, sale - saleCost - (acquisition - cum(S.depreciation, h))) * (h > 5 ? SALE_TAX_LONG : SALE_TAX_SHORT) / 100 : 0;
    return { sale, saleCost, bal, saleTax, net: sale - saleCost - bal - saleTax };
  };
  const scenario = (S, h, c) => {
    const { sale, saleCost, bal, saleTax, net } = saleNet(S, h, c);
    const f = [-initial, ...S.cf.slice(0, h)]; f[h] += net;
    return { total: -initial + cum(S.cf, h) + net, irr: irr(f), sale_price: sale, sale_cost: saleCost, balance: bal, sale_tax: saleTax, cash_flow: cum(S.cf, h),
      income_tax: tax ? cum(S.income_tax, h) : 0,
      ...Object.fromEntries(['rent', 'management', 'repair', 'rental_mgmt', 'other', 'turnover', 'interest', 'principal'].map(k => [k, cum(S[k], h)])) };
  };
  const priceChanges = SIM_PRICE_OFFSETS.map(o => price_change_pct + o);
  const grid = SIM_HOLD_YEARS.map(h => priceChanges.map(c => {
    const b = scenario(B, h, c);
    return { ...scenario(Y, h, c), total_base: b.total, irr_base: b.irr };
  }));
  // 売却する年ごと(1〜SIM_YEARS年後)の総収支 = 保有中の収支(−自己資金 + 収支の累計) + 売却代金の手取り。価格の列ごと
  const trend = priceChanges.map(c => {
    const o = { holding: [], sale_net: [], total: [], total_base: [] };
    let cf = -initial, cfB = -initial;
    for (let h = 1; h <= SIM_YEARS; h++) {
      cf += Y.cf[h - 1]; cfB += B.cf[h - 1];
      const net = saleNet(Y, h, c).net;
      o.holding.push(cf); o.sale_net.push(net); o.total.push(cf + net); o.total_base.push(cfB + saleNet(B, h, c).net);
    }
    return o;
  });
  return { price, loan_amount: loan, initial: { total: initial, costs: initial - init.down, items: init },
    running: { month: sum(Object.values(run)), items: run }, turnover: { per_turnover: perTurn, year: perTurn * units / t.interval_years, items: tv }, units,
    risk, tax: taxInfo, yearly: Y, yearly_base: B, hold_years: SIM_HOLD_YEARS, price_changes: priceChanges, grid, trend };
}

// 人口: 年齢層は5歳階級の下限で選ぶ(90 = 90歳以上)
export const AGE_GROUPS = { all: ['総数', 0, 90], child: ['0〜14歳', 0, 10], work: ['15〜64歳', 15, 60], young: ['20〜39歳', 20, 35],
  elderly: ['65歳以上', 65, 90], old75: ['75歳以上', 75, 90] };
const SEXES = { all: '男女計', m: '男', f: '女' };

/** 地域・性別・年齢層の各年の人口。値が欠けた年は null。 */
function popSeries(P, code, sex = 'all', group = 'all') {
  const a = P.pop[code], na = P.ages.length;
  if (!a) throw new Error(`人口のデータがありません: ${code}`);
  if (!AGE_GROUPS[group] || !SEXES[sex]) throw new Error(`年齢層・性別が不正です: ${group} ${sex}`);
  const [, lo, hi] = AGE_GROUPS[group];
  return P.years.map((_, yi) => {
    let t = 0;
    for (const s of sex === 'all' ? ['m', 'f'] : [sex]) for (let ai = 0; ai < na; ai++) {
      if (P.ages[ai] < lo || P.ages[ai] > hi) continue;
      const v = a[s][yi * na + ai];
      if (v == null) return null;
      t += v;
    }
    return t;
  });
}

/** 1地域の推移: 男女別・年齢層別の人口と構成比、人口ピラミッド(5歳階級×男女、年ごと)。 */
export async function populationArea({ code }) {
  const P = await loadPopulation(), na = P.ages.length, a = P.pop[code];
  if (!a) throw new Error(`人口のデータがありません: ${code}`);
  const total = popSeries(P, code), groups = {};
  for (const g of ['child', 'work', 'elderly', 'old75']) groups[g] = popSeries(P, code, 'all', g);
  const share = Object.fromEntries(Object.entries(groups).map(([g, v]) => [g, v.map((x, i) => x == null || !total[i] ? null : x / total[i])]));
  const pyramid = { m: P.years.map((_, yi) => a.m.slice(yi * na, yi * na + na)), f: P.years.map((_, yi) => a.f.slice(yi * na, yi * na + na)) };
  // 10歳階級(90歳以上はひとつ)の推移。[階級][年]、欠けた年は null
  const los = [...new Set(P.ages.map(x => Math.min(Math.floor(x / 10) * 10, 90)))];
  const dec = s => los.map(lo => P.years.map((_, yi) => {
    let t = 0;
    for (let ai = 0; ai < na; ai++) {
      if (Math.min(Math.floor(P.ages[ai] / 10) * 10, 90) !== lo) continue;
      const v = a[s][yi * na + ai];
      if (v == null) return null;
      t += v;
    }
    return t;
  }));
  const dm = dec('m'), df = dec('f');
  const decades = { labels: los.map(lo => lo === 90 ? '90歳以上' : `${lo}〜${lo + 9}歳`), m: dm, f: df,
    all: dm.map((r, i) => r.map((v, j) => v == null || df[i][j] == null ? null : v + df[i][j])) };
  return { code, name: P.names[code], years: P.years, ages: P.ages, actual_last: P.actual_last, total,
    male: popSeries(P, code, 'm'), female: popSeries(P, code, 'f'), groups, share, pyramid, decades,
    age_groups: Object.fromEntries(Object.entries(AGE_GROUPS).map(([k, v]) => [k, v[0]])), sexes: SEXES };
}

/** 主要都市(と全国)の比較: 選んだ性別・年齢層の人口と、基準年=100の指数。 */
export async function populationCompare({ sex = 'all', group = 'all', base = 2020 }) {
  const P = await loadPopulation(), bi = P.years.indexOf(base);
  if (bi < 0) throw new Error(`基準年が不正です: ${base}`);
  const rows = [P.nation, ...P.major].map(code => {
    const v = popSeries(P, code, sex, group), tot = popSeries(P, code), last = v.length - 1;
    return { code, name: P.names[code], values: v, index: v.map(x => x == null || !v[bi] ? null : x / v[bi] * 100),
      base_value: v[bi], last_value: v[last], change: v[bi] && v[last] != null ? v[last] / v[bi] - 1 : null, share_last: tot[last] ? v[last] / tot[last] : null };
  });
  return { years: P.years, base, sex, group, label: `${SEXES[sex]}・${AGE_GROUPS[group][0]}`, rows };
}

/** 地図の値: metric 'change' = from年→to年の増減率、'share' = year年の総人口に占める割合。性別・年齢層で絞れる。 */
export async function populationMap({ metric = 'change', from = 2020, to = 2050, year = 2050, sex = 'all', group = 'all' }) {
  const P = await loadPopulation(), fi = P.years.indexOf(from), ti = P.years.indexOf(to), yi = P.years.indexOf(year);
  if (metric === 'change' ? fi < 0 || ti < 0 || from >= to : yi < 0) throw new Error('地図の年の指定が不正です。');
  if (metric === 'share' && group === 'all') throw new Error('割合は年齢層を選んで表示します。');
  const values = {};
  for (const code of [...P.map_areas, ...P.prefs, P.nation]) {
    const v = popSeries(P, code, sex, group);
    if (metric === 'change') values[code] = v[fi] && v[ti] != null ? v[ti] / v[fi] - 1 : null;
    else { const t = popSeries(P, code)[yi]; values[code] = v[yi] != null && t ? v[yi] / t : null; }
  }
  return { metric, from, to, year, sex, group, values, nation: values[P.nation], names: P.names, prefs: P.prefs, map_areas: P.map_areas };
}

/** 地図の境界({コード: [[[経度, 緯度], …], …]})。 */
export async function populationGeo() {
  return loadBoundaries();
}

/** 長期推移: 都市の地価(地価公示)・家賃(消費者物価)・物価と、市街地価格指数・不動産価格指数(マンション)を base年=100 にそろえる。
 *  real なら全国の物価(持家の帰属家賃を除く総合)で割った実質値。基準年に値のない系列と、物価の値がない年(最新年など)の実質値は null。
 *  ratio = 地価 ÷ 家賃(base年=100。名目・実質で同じ)。都市の比較は metric('land' | 'rent' | 'ratio')の線と、基準年→最新の倍率の表。 */
export async function longTerm({ city, base = 1985, real = false, metric = 'land' }) {
  const [L, R] = await Promise.all([loadLong(), loadRates()]);
  if (!L.land[city]) throw new Error(`長期推移の都市が不正です: ${city}`);
  const bi = L.years.indexOf(base);
  if (bi < 0) throw new Error(`基準年が不正です: ${base}`);
  if (!['land', 'rent', 'ratio'].includes(metric)) throw new Error(`比較の指標が不正です: ${metric}`);
  const cpi = L.nation.cpi;
  const nominal = v => v[bi] == null ? null : v.map(x => x == null ? null : x / v[bi] * 100);
  const deflate = v => !v || !real ? v : cpi[bi] == null ? null : v.map((x, i) => x == null || cpi[i] == null ? null : x / (cpi[i] / cpi[bi]));
  const rebase = v => deflate(nominal(v));
  const ratioOf = c => nominal(L.land[c].map((x, i) => x == null || L.rent[c][i] == null ? null : x / L.rent[c][i]));
  // 基準年に値がない価格の系列(マンションの価格指数・東京区部の市街地価格指数)は、最初の年を地価(公示)の値に合わせて描く
  const landN = nominal(L.land[city]), anchored = {};
  const price = (k, v) => {
    let n = nominal(v);
    if (!n && landN) {
      const i = v.findIndex((x, j) => x != null && landN[j] != null);
      if (i >= 0) { n = v.map(x => x == null ? null : x / v[i] * landN[i]); anchored[k] = L.years[i]; }
    }
    return deflate(n);
  };
  // 不動産価格指数(月次)の年平均。12か月そろう年だけ
  const region = R.rpi_region[city], months = R.month, mansion = L.years.map(() => null);
  if (region) {
    const sum = {}, cnt = {};
    R.rpi[region].mansion.forEach((v, i) => { if (v != null) { const y = +months[i].slice(0, 4); sum[y] = (sum[y] ?? 0) + v; cnt[y] = (cnt[y] ?? 0) + 1; } });
    L.years.forEach((y, i) => { if (cnt[y] === 12) mansion[i] = sum[y] / 12; });
  }
  const series = {
    land: rebase(L.land[city]), rent: rebase(L.rent[city]), cpi: real ? null : nominal(cpi),
    jrei_six: rebase(L.nation.jrei_six), jrei_tokyo: price('jrei_tokyo', L.nation.jrei_tokyo), jrei_nation: rebase(L.nation.jrei_nation),
    mansion: price('mansion', mansion),
  };
  const last = v => v ? v.findLastIndex(x => x != null) : -1;
  const change = v => { const i = last(v); return i < 0 ? null : { year: L.years[i], value: v[i] / 100 }; };
  // 地価のピーク(基準年によらない。実質なら物価で割った値で探す)と、最新年のピークに対する比
  const peak = c => {
    const v = L.land[c].map((x, i) => x == null || (real && cpi[i] == null) ? null : x / (real ? cpi[i] : 1));
    let p = -1; v.forEach((x, i) => { if (x != null && (p < 0 || x > v[p])) p = i; });
    const i = last(v);
    return { year: L.years[p], latest_vs_peak: v[i] / v[p] };
  };
  const rows = L.cities.map(c => {
    const land = rebase(L.land[c]), rent = rebase(L.rent[c]), ratio = ratioOf(c);
    return { city: c, land: change(land), rent: change(rent), ratio: change(ratio), land_peak: peak(c),
      line: { land, rent, ratio }[metric] };
  });
  const li = L.years.indexOf(L.land_last_year);
  return {
    city, base, real, metric, years: L.years, series, anchored, ratio: ratioOf(city), rpi_region: region ?? null,
    land_n: L.land_n[city], land_last_year: L.land_last_year, rent_survey_year: L.rent_survey_year,
    // 円の目安(名目): 地価は最新年の地点平均を指数で過去へ、家賃は住宅・土地統計調査の年の平均を指数で前後へ延ばす
    land_yen: L.land[city].map(x => x == null ? null : L.land_level[city] * x / L.land[city][li]),
    rent_yen: L.rent_level[city] == null ? null : (() => { const si = L.years.indexOf(L.rent_survey_year), s = L.rent[city][si];
      return s == null ? null : L.rent[city].map(x => x == null ? null : L.rent_level[city] * x / s); })(),
    summary: { land: change(series.land), rent: change(series.rent), cpi: change(nominal(cpi)), ratio: change(ratioOf(city)), land_peak: peak(city) },
    rows,
  };
}

const LONG_YIELD_STD = { area: 25, age: 20, station_min: 10 };  // 標準的なワンルーム
/** 長期推移: 標準的なワンルーム(25㎡・築20年・駅徒歩10分)の表面利回りの推計。成約物件の実際の利回りではない。
 *  回帰の期間(2010年〜): 区ごとに 想定家賃 × 12 ÷ 回帰の推定価格 を取引件数で加重平均。家賃は住宅・土地統計調査の水準を、
 *  消費者物価の家賃指数で各年に延ばす(指数のない最新年は最後の値のまま)。
 *  それより前: 回帰の最初の年の利回りから 家賃の変化 ÷ 物件価格の変化 でさかのぼる。物件価格の変化は
 *  s × 地価の変化 + (1 − s) × 物価の変化(建物分) とし、s = 1(地価と同じ率)と s = 0.5(半分は物価)の2通りを幅として返す。
 *  s を回帰の期間に合わせて推定しないのは、2010年以降は建築費の上昇などで物件価格が地価より大きく上がり、s が0か1に張り付くため。 */
export async function longYield({ city }) {
  const [L, m, loan] = await Promise.all([loadLong(), loadMeta(), loadLoan()]);
  if (!L.land[city]) throw new Error(`長期推移の都市が不正です: ${city}`);
  const f = await getFit({ city, kind: 'mansion' }), { ds } = f;
  property(f, m, LONG_YIELD_STD);  // 区ごとの行(f.byWard)を作る
  const yi = y => L.years.indexOf(y), rentIdx = L.rent[city], lastRent = rentIdx.findLastIndex(v => v != null);
  const rentAt = y => rentIdx[Math.min(yi(y), lastRent)];
  const survey = rentAt(L.rent_survey_year);
  // 家賃指数のない年(相模原市は2015年〜など)は除く
  const years = [...new Set(Array.from(f.rows, r => year(ds, r)))].filter(y => rentAt(y) != null).sort((a, b) => a - b);
  if (!years.length) throw new Error(`${city} の家賃指数がありません。`);
  const wards = [];
  for (const [w, rows] of f.byWard) {
    let rent;
    try { rent = estimatedRent(m, ds.wards[w], city, LONG_YIELD_STD.area, LONG_YIELD_STD.age).rent; } catch { continue; }  // その区・面積区分の家賃統計がない
    wards.push({ n: rows.length, rent, prop: property(f, m, { ...LONG_YIELD_STD, ward: ds.wards[w] }).prop });
  }
  if (!wards.length) throw new Error(`${city} の家賃統計がありません。`);
  const n = sum(wards.map(w => w.n));
  const est = years.map(y => {
    let price = 0, rent = 0, gross = 0;
    for (const w of wards) {
      const pr = Math.exp(predict(f, { ...w.prop, year: y }).mu), rt = w.rent * rentAt(y) / survey;
      price += w.n * pr; rent += w.n * rt; gross += w.n * rt * 12 / pr;
    }
    return { price: price / n, rent: rent / n, gross: gross / n };
  });
  // 回帰の最初の年 y0 を起点に、地価(と物価)で価格をさかのぼる
  const y0 = years[0], land = L.land[city], cpi = L.nation.cpi, i0 = yi(y0);
  const mix = (s, i) => s * land[i] / land[i0] + (1 - s) * cpi[i] / cpi[i0];
  const back = { years: [], land: [], half: [] };
  L.years.forEach((y, i) => {
    if (y > y0 || land[i] == null || cpi[i] == null || rentIdx[i] == null) return;
    const g = est[0].gross * rentIdx[i] / rentAt(y0);
    back.years.push(y); back.land.push(g / mix(1, i)); back.half.push(g / mix(0.5, i));
  });
  // 比べる金利の年平均: 10年国債、変動金利の店頭の目安(短プラ+1%)、投資用の目安として住宅金融支援機構の賃貸住宅融資(35年固定)。
  // minMonths: 年平均に使う最少の月数(賃貸住宅融資は2006年4月からなので、ある月だけで平均する)
  const annual = (months, values, minMonths) => {
    const acc = {};
    months.forEach((mo, i) => { const v = values[i]; if (v != null) (acc[mo.slice(0, 4)] ??= []).push(v); });
    const a = Object.entries(acc).filter(([, v]) => v.length >= minMonths).sort(([x], [y]) => x - y);
    return { years: a.map(([y]) => +y), values: a.map(([, v]) => sum(v) / v.length) };
  };
  const r = L.rates, series = k => annual(r.month, r.values[k], 12), prime = series('prime_short');
  return { city, standard: LONG_YIELD_STD, years, price: est.map(e => e.price), rent: est.map(e => e.rent), gross: est.map(e => e.gross),
    back, rent_last_year: L.years[lastRent], rent_survey_year: L.rent_survey_year,
    rates: { jgb10: series('jgb10'), float_store: { years: prime.years, values: prime.values.map(v => v + 1) },
      chintai_35: annual(loan.chintai.month, loan.chintai.rates.free_35, 1) } };
}

/** 金利の長期推移(月次、1882年〜)。変動の店頭金利の目安(短プラ+1%)と、物価上昇率(全国の持家の帰属家賃を除く総合の前年比、年次)も返す。 */
export async function longRates() {
  const L = await loadLong(), r = L.rates, cpi = L.nation.cpi;
  return { month: r.month, labels: { ...r.labels, float_store: '変動金利の店頭の目安(短プラ+1%)' },
    values: { ...r.values, float_store: r.values.prime_short.map(v => v == null ? null : v + 1) },
    inflation: { years: L.years.slice(1), values: L.years.slice(1).map((_, i) => cpi[i] == null || cpi[i + 1] == null ? null : (cpi[i + 1] / cpi[i] - 1) * 100) } };
}

export const routes = { meta, fit: fitSummary, profile, forecast, yield: investmentYield, rates, districts, 'district-map': districtMap, population: populationSeries, market,
  'housing-market': housingMarket, 'housing-national': housingNational, loan: loanOverview, repayment, 'loan-calc': loanCalc, cashflow: cashflowSim,
  long: longTerm, 'long-rates': longRates, 'long-yield': longYield, 'population-area': populationArea, 'population-compare': populationCompare, 'population-map': populationMap, 'population-geo': populationGeo };
