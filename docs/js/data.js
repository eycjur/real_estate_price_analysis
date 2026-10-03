// docs/data/ の読み込み。取引データは 都市×種別ごとの列指向バイナリ(gzip) で、reap.export が書き出す。

let loadBytes = async path => {
  const r = await fetch(new URL(`../data/${path}`, import.meta.url));
  if (!r.ok) throw new Error(`データを読み込めません: ${path} (${r.status})`);
  return r.arrayBuffer();
};
/** テスト用: データの読み込み方を差し替える。fn(path) → ArrayBuffer */
export function setLoader(fn) { loadBytes = fn; cache.clear(); metaPromise = null; ratesPromise = null; marketPromise = null; loanPromise = null; longPromise = null; popPromise = null; geoPromise = null; }

const TYPES = { uint8: Uint8Array, uint16: Uint16Array, uint32: Uint32Array };
const cache = new Map();
let metaPromise = null;

const loadJson = async path => JSON.parse(new TextDecoder().decode(await loadBytes(path)));
async function gunzip(buf) {
  return new Response(new Blob([buf]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
}

export function loadMeta() {
  return metaPromise ??= loadJson('meta.json');
}

let ratesPromise = null;
/** 金利と不動産価格指数の月次系列(docs/data/rates.json)。 */
export function loadRates() {
  return ratesPromise ??= loadJson('rates.json');
}

let marketPromise = null;
/** 市況(レインズ・既存住宅販売量指数・着工統計)の系列(docs/data/market.json)。 */
export function loadMarket() {
  return marketPromise ??= loadJson('market.json');
}

let loanPromise = null;
/** ローン(機構の賃貸住宅融資の金利・日銀の新規貸出額・商品の金利・金利タイプ別の利用割合)(docs/data/loan.json)。 */
export function loadLoan() {
  return loanPromise ??= loadJson('loan.json');
}

let longPromise = null;
/** 長期推移(地価公示の地価指数・消費者物価の家賃指数・全国の物価・市街地価格指数の年次系列)(docs/data/long.json)。 */
export function loadLong() {
  return longPromise ??= loadJson('long.json');
}

let popPromise = null, geoPromise = null;
/** 男女・5歳階級別の人口(docs/data/population.json)と、地図用の市区町村の境界(boundaries.json)。 */
export function loadPopulation() {
  return popPromise ??= loadJson('population.json');
}
export function loadBoundaries() {
  return geoPromise ??= loadJson('boundaries.json');
}

/** 取引データに、回帰で毎回使う対数などの派生列を足す。 */
function finish(ds) {
  const c = ds.cols, n = ds.n;
  ds.lnPrice = new Float64Array(n); ds.lnArea = new Float64Array(n);
  for (let i = 0; i < n; i++) { ds.lnPrice[i] = Math.log(c.price[i] * 10000); ds.lnArea[i] = Math.log(c.area[i]); }
  if (c.land_area) { ds.lnLand = new Float64Array(n); for (let i = 0; i < n; i++) ds.lnLand[i] = Math.log(c.land_area[i]); }
  return ds;
}

/** 1都市×1種別の取引データ。cols の year は西暦−2000、price は万円、カテゴリ列は meta.levels の番号。 */
export function loadDataset(kind, city) {
  const key = `${kind}|${city}`;
  if (!cache.has(key)) cache.set(key, (async () => {
    const meta = await loadMeta();
    const entry = meta.datasets[kind]?.[city];
    if (!entry) throw new Error(`データがありません: ${city} / ${kind}`);
    const [info, gz] = await Promise.all([loadJson(`${entry.file}.json`), loadBytes(`${entry.file}.bin.gz`)]);
    const buf = await gunzip(gz), cols = {};
    for (const col of info.columns) cols[col.name] = new TYPES[col.dtype](buf, col.offset, info.n);
    return finish({ kind, n: info.n, cities: [city], cityOf: new Uint8Array(info.n), wards: info.wards,
      wardCity: info.wards.map(() => 0), districts: info.districts, cols });
  })());
  return cache.get(key);
}

/** 全都市を連結したデータ(区・地区の番号は都市ごとにずらして通し番号にする)。 */
export function loadAll(kind) {
  const key = `${kind}|*`;
  if (!cache.has(key)) cache.set(key, (async () => {
    const meta = await loadMeta();
    const cities = meta.cities.filter(c => meta.datasets[kind][c]);
    const parts = await Promise.all(cities.map(c => loadDataset(kind, c)));
    const n = parts.reduce((t, p) => t + p.n, 0);
    const all = { kind, n, cities, cityOf: new Uint8Array(n), wards: [], wardCity: [], districts: [], cols: {} };
    for (const name of Object.keys(parts[0].cols)) {
      const wide = name === 'ward' || name === 'district';  // 通し番号にすると元の型に収まらない
      all.cols[name] = new (wide ? Uint32Array : parts[0].cols[name].constructor)(n);
    }
    let off = 0;
    parts.forEach((p, ci) => {
      const w0 = all.wards.length, d0 = all.districts.length;
      for (const [name, a] of Object.entries(p.cols)) {
        if (name === 'ward') for (let i = 0; i < p.n; i++) all.cols.ward[off + i] = a[i] + w0;
        else if (name === 'district') for (let i = 0; i < p.n; i++) all.cols.district[off + i] = a[i] === meta.no_district ? meta.no_district : a[i] + d0;
        else all.cols[name].set(a, off);
      }
      all.cityOf.fill(ci, off, off + p.n);
      all.wards.push(...p.wards); all.wardCity.push(...p.wards.map(() => ci));
      all.districts.push(...p.districts.map(d => ({ ...d, ward: d.ward + w0 })));
      off += p.n;
    });
    return finish(all);
  })());
  return cache.get(key);
}
