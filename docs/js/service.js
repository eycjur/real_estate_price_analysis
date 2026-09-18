// 画面から呼ぶ分析処理 (モデル推定のキャッシュ、断面プロファイル、将来予測、利回り、相場)。
// Web Worker (worker.js) の中で動く。

import { loadAll, loadDataset, loadMeta } from './data.js';
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
    ref: { age: ref.ref_age, area: ref.ref_area, land: ref.ref_land ?? 1, station: ref.ref_station } };
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
  return { cities: m.cities.map(c => infos[c]), kinds: m.kinds, all_label: m.all_label, year_min: m.year_min, year_max: m.year_max, n_total: m.n_total };
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
  const base = { age: +age.levels[0], area: s.ref.area, station_min: s.ref.station, ward: baseWard, year: baseYear ?? maxYear,
    price: Math.exp(pooled ? coef(f, `ward=${baseWard}`).coef : coef(f, 'const').coef) };
  if (f.design.house) base.land_area = s.ref.land;

  const shown = sample(rows.map((_, i) => i), PRED_ACTUAL_POINTS, 0);
  const cats = {};
  for (const v of f.design.vars) if (m.variables[p.kind].includes(v.name)) cats[v.name] = { label: m.labels[v.name], levels: v.levels };
  return {
    params: { city: p.city, kind: p.kind, year_from: p.year_from ?? 2010, year_to: p.year_to ?? 2100 },
    n, r2: f.r2, adj_r2: f.adjR2, rmse: f.rmse, mae: f.mae, typical_error_pct: Math.expm1(f.rmse), se_type: f.seType,
    within_20pct: within20 / n, within_1sd: within1sd / n, base,
    pred_actual: { pred: shown.map(i => Math.round(Math.exp(ds.lnPrice[rows[i]] - resid[i]) / 1000) * 1000), actual: shown.map(i => yen(ds, rows[i])) },
    coefficients: rowsTable.filter(r => !FE_PREFIX.some(x => r.name.startsWith(x))), base_age: +age.levels[0],
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
  if (kind === 'house') view.land_area = +(prop.land_area || median(Array.from(rows, r => ds.cols.land_area[r])));
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
  area: ['建物面積(㎡)', a => linspace(Math.max(15, quantile(a.area, 0.01)), quantile(a.area, 0.99), 50)],
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
  const grid = gridOf({ get area() { return column('area'); }, get land_area() { return column('land_area'); }, get year() { return column('year'); } });
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
    && (vary === 'land_area' || ds.kind !== 'house' || ratio(c.land_area[r], view.land_area))
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
function yields(price, rent, c) {
  const income = rent * 12 * (1 - c.vacancy_pct / 100);
  const noi = income * (1 - c.management_fee_pct / 100) - c.building_fee_month * 12 - c.property_tax_year;
  return { gross: rent * 12 / price, net: noi / (price * (1 + c.purchase_cost_pct / 100)), noi };
}

/** 回帰モデルの推定価格と統計ベースの想定家賃から、表面利回りと実質(NOI)利回りを出す。 */
export async function investmentYield({ fit: p, property: input, costs = {} }) {
  if (p.kind !== 'mansion') throw new Error('利回りの試算は中古マンションのみ対応しています。');
  const unknown = Object.keys(costs).filter(k => !(k in COSTS));
  if (unknown.length) throw new Error(`利回りの前提が不正です: ${unknown}`);
  const c = { ...COSTS, ...costs }, f = await getFit(p), m = await loadMeta(), { view, prop } = property(f, m, input);
  const rentAt = (ward, age) => c.rent_month ? { rent: +c.rent_month, info: null } : estimatedRent(m, ward, view.city, view.area, age);
  const curve = { age: [], price: [], rent: [], gross: [], net: [] };
  let at = null, info = null;
  for (let age = 0; age <= 50; age++) {
    const price = Math.exp(predict(f, { ...prop, age }).mu), r = rentAt(view.ward, age), y = yields(price, r.rent, c);
    curve.age.push(age); curve.price.push(price); curve.rent.push(r.rent); curve.gross.push(y.gross); curve.net.push(y.net);
    if (age === Math.max(0, Math.min(50, Math.round(view.age)))) { at = { price, rent: r.rent, ...y }; info = r.info; }
  }
  const wards = [];
  for (const v of f.design.vars.find(x => x.name === 'ward').levels) {
    let r;
    try { r = rentAt(v, view.age); } catch { continue; }  // その区・面積区分の家賃統計がない
    const price = Math.exp(predict(f, property(f, m, input, { ward: v, district: null }).prop).mu);
    wards.push({ ward: v, price, rent: r.rent, ...yields(price, r.rent, c) });
  }
  return { ward: view.ward, area: view.area, age: view.age, year: view.year, ...at, rent_info: info, rent_overridden: !!c.rent_month, costs: c,
    curve, wards: wards.sort((a, b) => b.net - a.net) };
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

export const routes = { meta, fit: fitSummary, profile, forecast, yield: investmentYield, districts, 'district-map': districtMap, population: populationSeries, market };
