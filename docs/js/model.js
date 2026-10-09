// ヘドニック回帰 (目的変数: ln 取引価格)。src/reap/model.py と同じ推定を、ブラウザで動く形にしたもの。
//
// 説明変数はほとんどがダミー変数なので、計画行列は作らない。1行ごとに
// 「数値列の値」と「立っているダミーの列番号」だけをたどって X'X に足し込む。
// 地区(町名)効果は L2 罰則つきで吸収する: min ||y − Xβ − Dγ||² + λ||γ||²
// D'D が対角なので、地区別の合計だけでシューア補行列を作れば厳密解になる。

import { cholInverse, cholSolve, cholesky, independentColumns, matmul } from './linalg.js';
import { erfc } from './stats.js';

export const FE_PREFIX = ['ward=', 'year=', '_cy='];
const FE_LABEL = { age: '築年数', area: '建物面積', station: '駅徒歩', ward: '区', year: '取引年', _cy: '都市×年' };
const STATION_CAP = 30;  // 駅徒歩の区間は この分数以上を1区間にまとめる(元データも30分以上は幅でしかない)

/** 都市×年のコード(都市番号×32 + 西暦−2000)。全都市プールの固定効果用。 */
function cityYear(ds) {
  if (!ds.cy) {
    ds.cy = new Uint16Array(ds.n);
    for (let i = 0; i < ds.n; i++) ds.cy[i] = ds.cityOf[i] * 32 + ds.cols.year[i];
  }
  return ds.cy;
}

/**
 * spec: { pooled, lambda, ref: {age, area, areaStep, land, station, stationStep}, vars: [カテゴリ変数], baseLevels: {変数: 水準名} }
 * pooled=false は 区FE+年FE(都市内)、true は 区FE+都市×年FE(全都市プール、定数項なし)。
 * ref.areaStep を指定すると面積を step㎡ 刻みのカテゴリ変数(基準 = ref.area を含む区分)にし、なければ ln 面積で入れる。
 * ref.stationStep を指定すると駅徒歩分を step分 刻みの区間(30分以上は1区間、基準 = ref.station を含む区間)にし、なければ1次の項で入れる。
 */
export function buildDesign(ds, rows, spec, meta) {
  const house = !!ds.lnLand, step = spec.ref.areaStep, sstep = spec.ref.stationStep;
  const numeric = [...(spec.pooled ? [] : ['const']), ...(step ? [] : ['ln_area']), ...(house ? ['ln_land'] : []), ...(sstep ? [] : ['station_min'])];
  const numericLabel = { const: '定数項', ln_area: `ln(建物面積㎡/${spec.ref.area})`, ln_land: `ln(土地面積㎡/${spec.ref.land})`, station_min: `(駅徒歩分−${spec.ref.station})` };
  const names = [...numeric], labels = numeric.map(n => numericLabel[n]);

  const levelName = (v, code) => v === 'age' ? String(code) : v === 'area' ? String(code * step) : v === 'station' ? String(code * sstep) : v === 'ward' ? ds.wards[code] : v === 'year' ? String(2000 + code)
    : v === '_cy' ? `${ds.cities[code >> 5]}|${2000 + (code & 31)}` : meta.levels[ds.kind][v][code];
  const counts = (codes, size) => { const c = new Float64Array(size); for (const r of rows) c[codes[r]]++; return c; };
  const byFreq = c => [...c.keys()].filter(i => c[i] > 0).sort((a, b) => c[b] - c[a] || a - b);

  const vars = [];
  const add = (name, codes, size, levels, hasBase) => {
    const map = new Int32Array(size).fill(-1);
    const shown = code => name === 'age' ? `${code}年` : name === 'area' ? `${code * step}〜${(code + 1) * step}㎡`
      : name === 'station' ? (code * sstep >= STATION_CAP ? `${code * sstep}分以上` : `${code * sstep}〜${(code + 1) * sstep - 1}分`) : levelName(name, code);
    const base = hasBase ? shown(levels[0]) : null;
    for (const code of levels.slice(hasBase ? 1 : 0)) {
      map[code] = names.length;
      names.push(`${name}=${levelName(name, code)}`);
      labels.push(`${FE_LABEL[name] ?? meta.labels[name]}: ${shown(code)}${base == null ? '' : ` (基準: ${base})`}`);
    }
    vars.push({ name, codes, map, levels: levels.map(c => levelName(name, c)), hasBase });
  };

  // 築年数: 1年刻み。基準は ref.age 年(データになければ最も多い築年数)
  const ageCount = counts(ds.cols.age, 256), present = [...ageCount.keys()].filter(a => ageCount[a] > 0);
  const baseAge = ageCount[spec.ref.age] > 0 ? spec.ref.age : byFreq(ageCount)[0];
  add('age', ds.cols.age, 256, [baseAge, ...present.filter(a => a !== baseAge)], true);
  nearestLevel(vars[0], present);
  // 面積: step㎡ 刻みの区分(番号 = 面積÷step の切り捨て)。基準は ref.area を含む区分(データになければ最も多い区分)
  if (step) {
    const bins = Uint16Array.from(ds.cols.area, a => Math.floor(a / step));
    const size = Math.floor(65535 / step) + 1, binCount = counts(bins, size), bp = [...binCount.keys()].filter(b => binCount[b] > 0);
    const refBin = Math.floor(spec.ref.area / step), baseBin = binCount[refBin] > 0 ? refBin : byFreq(binCount)[0];
    add('area', bins, size, [baseBin, ...bp.filter(b => b !== baseBin)], true);
    nearestLevel(vars.at(-1), bp);
  }
  // 駅徒歩: sstep分 刻みの区間(番号 = 分÷sstep の切り捨て、30分以上は1区間)。基準は ref.station を含む区間
  if (sstep) {
    const top = Math.floor(STATION_CAP / sstep), bins = Uint8Array.from(ds.cols.station_min, m => Math.min(Math.floor(m / sstep), top));
    const binCount = counts(bins, top + 1), bp = [...binCount.keys()].filter(b => binCount[b] > 0);
    const refBin = Math.min(Math.floor(spec.ref.station / sstep), top), baseBin = binCount[refBin] > 0 ? refBin : byFreq(binCount)[0];
    add('station', bins, top + 1, [baseBin, ...bp.filter(b => b !== baseBin)], true);
    nearestLevel(vars.at(-1), bp);
  }
  for (const v of spec.vars) {
    const levels = byFreq(counts(ds.cols[v], meta.levels[ds.kind][v].length));
    const want = meta.levels[ds.kind][v].indexOf(spec.baseLevels?.[v]);
    if (levels.includes(want)) levels.splice(0, 0, ...levels.splice(levels.indexOf(want), 1));
    add(v, ds.cols[v], meta.levels[ds.kind][v].length, levels, true);
  }
  const wardLevels = byFreq(counts(ds.cols.ward, ds.wards.length));
  if (!spec.pooled) {
    add('ward', ds.cols.ward, ds.wards.length, wardLevels, true);
    const years = byFreq(counts(ds.cols.year, 64)).sort((a, b) => b - a);  // 最新年を基準に
    add('year', ds.cols.year, 64, years, true);
  } else {
    // 区ダミーは全水準(定数項なし)、都市×年は各都市の最新年を基準として落とす
    add('ward', ds.cols.ward, ds.wards.length, wardLevels, false);
    const cy = cityYear(ds), cyCount = counts(cy, ds.cities.length * 32);
    const latest = new Map();
    for (const c of cyCount.keys()) if (cyCount[c] > 0) latest.set(c >> 5, Math.max(latest.get(c >> 5) ?? 0, c & 31));
    add('_cy', cy, ds.cities.length * 32, [...cyCount.keys()].filter(c => cyCount[c] > 0 && (c & 31) !== latest.get(c >> 5)), false);
  }
  return { spec, house, numeric, names, labels, vars, dropped: [], baseWard: spec.pooled ? null : ds.wards[wardLevels[0]] };
}

/** 築年数・面積の区分の列番号を、推定に使った水準のうち最も近いものにする(データにない築年数や60年超などの予測用)。 */
function nearestLevel(v, present) {
  const exact = Int32Array.from(v.map);
  for (let a = 0; a < v.map.length; a++) {
    const near = present.reduce((best, p) => Math.abs(p - a) < Math.abs(best - a) ? p : best);
    v.map[a] = exact[near];
  }
}

/** 1行ぶんの説明変数。数値列の値を dv に、立っているダミーの列番号を ci に入れ、ダミーの個数を返す。 */
function fillRow(design, ds, r, dv, ci) {
  const s = design.spec;
  let j = 0;
  if (!s.pooled) dv[j++] = 1;
  if (!s.ref.areaStep) dv[j++] = ds.lnArea[r] - Math.log(s.ref.area);
  if (design.house) dv[j++] = ds.lnLand[r] - Math.log(s.ref.land);
  if (!s.ref.stationStep) dv[j++] = ds.cols.station_min[r] - s.ref.station;
  let nc = 0;
  for (const v of design.vars) { const c = v.map[v.codes[r]]; if (c >= 0) ci[nc++] = c; }
  return nc;
}

/** OLS(+地区効果の L2 吸収)。標準誤差は cluster=true なら区クラスタ頑健、そうでなければ HC1。 */
export function fit(ds, rows, spec, meta, { cluster = false } = {}) {
  const design = buildDesign(ds, rows, spec, meta);
  const d = design.numeric.length, lam = spec.lambda, NONE = meta.no_district, y = ds.lnPrice, dist = ds.cols.district;
  let k = design.names.length;
  const dv = new Float64Array(d), ci = new Int32Array(design.vars.length);
  let xtx = new Float64Array(k * k), xty = new Float64Array(k);
  const m = lam == null ? 0 : ds.districts.length;
  let Sx = new Float64Array(m * k);
  const sy = new Float64Array(m), nd = new Float64Array(m);  // 地区別の Σx, Σy, 件数

  for (const r of rows) {
    const nc = fillRow(design, ds, r, dv, ci), yr = y[r];
    for (let a = 0; a < d; a++) {
      for (let b = 0; b < d; b++) xtx[a * k + b] += dv[a] * dv[b];
      for (let b = 0; b < nc; b++) { xtx[a * k + ci[b]] += dv[a]; xtx[ci[b] * k + a] += dv[a]; }
      xty[a] += dv[a] * yr;
    }
    for (let a = 0; a < nc; a++) { for (let b = 0; b < nc; b++) xtx[ci[a] * k + ci[b]]++; xty[ci[a]] += yr; }
    if (m && dist[r] !== NONE) {
      const o = dist[r] * k;
      for (let a = 0; a < d; a++) Sx[o + a] += dv[a];
      for (let a = 0; a < nc; a++) Sx[o + ci[a]]++;
      sy[dist[r]] += yr; nd[dist[r]]++;
    }
  }
  const w = Float64Array.from(nd, n => 1 / (n + lam));
  const nonzero = (M, row, kk) => { const nz = []; for (let i = 0; i < kk; i++) if (M[row * kk + i] !== 0) nz.push(i); return nz; };
  for (let q = 0; q < m; q++) {  // シューア補行列: X'X − Sx' diag(w) Sx
    if (!nd[q]) continue;
    const nz = nonzero(Sx, q, k), o = q * k;
    for (const a of nz) { const sa = Sx[o + a] * w[q]; for (const b of nz) xtx[a * k + b] -= sa * Sx[o + b]; xty[a] -= sa * sy[q]; }
  }

  // 他の列と完全に重複する列を落とす(例: 戸建の「成約価格」「土地の形状=不明」「地域=不明」)
  const keep = independentColumns(xtx, k);
  if (keep.length < k) {
    const to = new Int32Array(k).fill(-1);
    keep.forEach((old, i) => { to[old] = i; });
    if (design.numeric.some((_, i) => to[i] < 0)) throw new Error('数値の説明変数が他の列と重複しています');
    design.dropped = design.labels.filter((_, i) => to[i] < 0);
    design.names = keep.map(i => design.names[i]); design.labels = keep.map(i => design.labels[i]);
    for (const v of design.vars) for (let c = 0; c < v.map.length; c++) if (v.map[c] >= 0) v.map[c] = to[v.map[c]];
    const k2 = keep.length, x2 = new Float64Array(k2 * k2), S2 = new Float64Array(m * k2);
    keep.forEach((a, i) => keep.forEach((b, j) => { x2[i * k2 + j] = xtx[a * k + b]; }));
    for (let q = 0; q < m; q++) keep.forEach((a, i) => { S2[q * k2 + i] = Sx[q * k + a]; });
    xty = Float64Array.from(keep, a => xty[a]); xtx = x2; Sx = S2; k = k2;
  }
  const L = cholesky(xtx, k), beta = cholSolve(L, k, xty), bread = cholInverse(L, k);

  // 実効自由度 = tr H − k = Σ n_d·w_d − λ Σ w_d²·Σx_d'(X'MX)⁻¹Σx_d。後ろの項は説明変数と地区効果の重複分
  let gamma = null, ddf = 0;
  if (m) {
    gamma = new Float64Array(m);
    for (let q = 0; q < m; q++) {
      let xb = 0;
      for (let a = 0; a < k; a++) xb += Sx[q * k + a] * beta[a];
      gamma[q] = (sy[q] - xb) * w[q];
      if (!nd[q]) continue;
      const nz = nonzero(Sx, q, k), o = q * k;
      let quad = 0;
      for (const a of nz) for (const b of nz) quad += Sx[o + a] * bread[a * k + b] * Sx[o + b];
      ddf += nd[q] * w[q] - lam * w[q] * w[q] * quad;
    }
  }

  // 残差と、頑健標準誤差の「肉」。地区効果を払い出した x̃ = x − w_d·Sx_d は密になるので、
  // Σe²x̃x̃' を Σe²xx'(疎) と地区ごとの集計 (Σe²x, Σe²) に展開して計算する
  const n = rows.length, resid = new Float64Array(n), meat = new Float64Array(k * k);
  const B = new Float64Array(cluster ? 0 : m * k), s2 = new Float64Array(m), E = new Float64Array(m);
  const G = ds.wards.length, Sg = new Float64Array(cluster ? G * k : 0), wardSeen = new Uint8Array(G);
  let sse = 0, sae = 0, sy1 = 0, sy2 = 0;
  rows.forEach((r, t) => {
    const nc = fillRow(design, ds, r, dv, ci), q = m && dist[r] !== NONE ? dist[r] : -1;
    let e = y[r];
    for (let a = 0; a < d; a++) e -= dv[a] * beta[a];
    for (let a = 0; a < nc; a++) e -= beta[ci[a]];
    if (q >= 0) e -= gamma[q];
    resid[t] = e; sse += e * e; sae += Math.abs(e); sy1 += y[r]; sy2 += y[r] * y[r];
    wardSeen[ds.cols.ward[r]] = 1;
    if (cluster) {
      const o = ds.cols.ward[r] * k;
      for (let a = 0; a < d; a++) Sg[o + a] += e * dv[a];
      for (let a = 0; a < nc; a++) Sg[o + ci[a]] += e;
      if (q >= 0) E[q] += e;
      return;
    }
    const e2 = e * e;
    for (let a = 0; a < d; a++) {
      for (let b = 0; b < d; b++) meat[a * k + b] += e2 * dv[a] * dv[b];
      for (let b = 0; b < nc; b++) { meat[a * k + ci[b]] += e2 * dv[a]; meat[ci[b] * k + a] += e2 * dv[a]; }
    }
    for (let a = 0; a < nc; a++) for (let b = 0; b < nc; b++) meat[ci[a] * k + ci[b]] += e2;
    if (q >= 0) {
      for (let a = 0; a < d; a++) B[q * k + a] += e2 * dv[a];
      for (let a = 0; a < nc; a++) B[q * k + ci[a]] += e2;
      s2[q] += e2;
    }
  });
  for (let q = 0; q < m; q++) {
    if (!nd[q]) continue;
    const nz = nonzero(Sx, q, k), o = q * k;
    if (cluster) { const g = ds.districts[q].ward * k; for (const a of nz) Sg[g + a] -= E[q] * w[q] * Sx[o + a]; continue; }
    for (const a of nz) {
      const wa = w[q] * Sx[o + a];
      for (const b of nz) { const wb = w[q] * Sx[o + b]; meat[a * k + b] += s2[q] * wa * wb - B[o + a] * wb - wa * B[o + b]; }
    }
  }
  let nG = 0;
  for (let g = 0; g < G; g++) {
    if (!wardSeen[g]) continue;
    nG++;
    if (cluster) for (let a = 0; a < k; a++) { const sa = Sg[g * k + a]; if (sa) for (let b = 0; b < k; b++) meat[a * k + b] += sa * Sg[g * k + b]; }
  }
  const p = k + ddf;  // 実効パラメータ数
  const adj = cluster ? (nG / (nG - 1)) * ((n - 1) / (n - p)) : n / (n - p);
  const cov = matmul(matmul(bread, meat, k), bread, k).map(v => v * adj);
  const r2 = 1 - sse / (sy2 - sy1 * sy1 / n);
  return { design, ds, rows, beta, cov, se: Float64Array.from({ length: k }, (_, i) => Math.sqrt(cov[i * k + i])), n, r2,
    adjR2: 1 - (1 - r2) * (n - 1) / (n - p), rmse: Math.sqrt(sse / (n - p)), mae: sae / n,
    seType: cluster ? 'cluster(区)' : 'HC1', gamma, districtDf: ddf, districtX: Sx, districtW: w, districtN: nd, resid };
}

export function coef(f, name) {
  const i = f.design.names.indexOf(name);
  return i < 0 ? null : { coef: f.beta[i], se: f.se[i] };
}

export function table(f, includeFE = true) {
  const out = [];
  f.design.names.forEach((name, i) => {
    if (!includeFE && FE_PREFIX.some(p => name.startsWith(p))) return;
    const b = f.beta[i], s = f.se[i], t = b / s;
    out.push({ name, label: f.design.labels[i], coef: b, se: s, t, p: erfc(Math.abs(t) / Math.SQRT2),  // n が大きいので正規近似
      ci_low: b - 1.96 * s, ci_high: b + 1.96 * s });
  });
  return out;
}

/** データの行(rows)に対する ln価格の予測値。推定に使っていない行(検証用)にも使える。 */
export function predictRows(f, rows) {
  const { design, ds, beta, gamma } = f, d = design.numeric.length, NONE = 0xFFFF;
  const dv = new Float64Array(d), ci = new Int32Array(design.vars.length), out = new Float64Array(rows.length);
  rows.forEach((r, t) => {
    const nc = fillRow(design, ds, r, dv, ci);
    let mu = 0;
    for (let a = 0; a < d; a++) mu += dv[a] * beta[a];
    for (let a = 0; a < nc; a++) mu += beta[ci[a]];
    const q = ds.cols.district[r];
    out[t] = mu + (gamma && q !== NONE ? gamma[q] : 0);  // 推定時にない地区の効果は 0 = 区の平均的な立地
  });
  return out;
}

/**
 * 任意の物件の ln価格の予測値と、その平均の標準誤差。
 * 地区を指定すると、γ_d = w_d(Σy_d − Σx_d'β) なので予測値 = (x − w_d·Σx_d)'β + w_d·Σy_d。分散は β の推定誤差
 * (x − w_d·Σx_d)'V(x − w_d·Σx_d) に、地区効果の推定誤差 σ²·w_d(L2 罰則を事前分布とみたときの事後分散)を足す。
 * prop: { age, area, land_area, station_min, ward(番号), year(西暦), city(番号), district(番号 | -1), cats: {変数: 水準番号} }
 */
export function predict(f, prop) {
  const { design, beta, cov, gamma } = f, s = design.spec, k = beta.length;
  const idx = [], val = [];
  const num = { const: 1, ln_area: Math.log(prop.area / s.ref.area), ln_land: Math.log(prop.land_area / s.ref.land), station_min: prop.station_min - s.ref.station };
  design.numeric.forEach((name, i) => { idx.push(i); val.push(num[name]); });
  for (const v of design.vars) {
    const code = v.name === 'age' ? Math.max(0, Math.min(255, Math.round(prop.age)))
      : v.name === 'area' ? Math.max(0, Math.min(v.map.length - 1, Math.floor(prop.area / s.ref.areaStep)))
      : v.name === 'station' ? Math.max(0, Math.min(v.map.length - 1, Math.floor(prop.station_min / s.ref.stationStep))) : v.name === 'ward' ? prop.ward
      : v.name === 'year' ? prop.year - 2000 : v.name === '_cy' ? prop.city * 32 + prop.year - 2000 : prop.cats[v.name];
    const c = v.map[code] ?? -1;
    if (c >= 0) { idx.push(c); val.push(1); }
  }
  const c = new Float64Array(k);
  idx.forEach((a, i) => { c[a] += val[i]; });
  let mu = 0, variance = 0;
  for (let a = 0; a < k; a++) mu += c[a] * beta[a];
  const q = gamma && prop.district >= 0 && f.districtN[prop.district] ? prop.district : -1;
  if (q >= 0) {
    mu += gamma[q];
    for (let a = 0; a < k; a++) c[a] -= f.districtW[q] * f.districtX[q * k + a];
    variance += f.rmse * f.rmse * f.districtW[q];
  }
  for (let a = 0; a < k; a++) if (c[a]) for (let b = 0; b < k; b++) if (c[b]) variance += c[a] * c[b] * cov[a * k + b];
  return { mu, se: Math.sqrt(variance) };
}

/** 築年数 a0→a1 による ln価格の変化とその標準誤差(築年数ダミーの係数の差)。 */
export function ageEffect(f, a0, a1) {
  const k = f.beta.length, age = f.design.vars[0], c = new Float64Array(k);
  for (const [a, sign] of [[a1, 1], [a0, -1]]) {
    const col = age.map[Math.max(0, Math.min(255, Math.round(a)))];
    if (col >= 0) c[col] += sign;  // 基準の築年数は係数0
  }
  let effect = 0, variance = 0;
  for (let a = 0; a < k; a++) if (c[a]) { effect += c[a] * f.beta[a]; for (let b = 0; b < k; b++) if (c[b]) variance += c[a] * c[b] * f.cov[a * k + b]; }
  return { effect, se: Math.sqrt(variance) };
}
