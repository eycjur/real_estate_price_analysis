// pytest から呼ばれる: 指定ディレクトリのデータで JS の回帰を実行し、結果を JSON で標準出力に出す。
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { loadAll, loadDataset, loadMeta, setLoader } from '../../docs/js/data.js';
import { ageEffect, fit, predict } from '../../docs/js/model.js';

const dir = process.argv[2];
setLoader(async p => { const b = await readFile(path.join(dir, p)); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); });
const meta = await loadMeta();
const spec = { lambda: 5, ref: { age: 10, area: 60, land: 1, station: 8 }, vars: Object.keys(meta.levels.mansion) /* dup を含む全列 */, baseLevels: {} };
const all = n => Int32Array.from({ length: n }, (_, i) => i);
const dump = f => ({ coef: Object.fromEntries(f.design.names.map((n, i) => [n, [f.beta[i], f.se[i]]])), r2: f.r2, adj_r2: f.adjR2, rmse: f.rmse, mae: f.mae,
  dropped: f.design.dropped, district_df: f.districtDf, gamma: Object.fromEntries(f.ds.districts.map((d, i) => [`${f.ds.wards[d.ward]} ${d.name}`, f.gamma[i]])) });

const ds = await loadDataset('mansion', 'X市');
const city = fit(ds, all(ds.n), { ...spec, pooled: false }, meta);
const prop = { age: 23, area: 72, station_min: 5, ward: ds.wards.indexOf('B区'), year: 2018, city: 0, district: 3, cats: { structure: 1, renovated: 0, dup: 0 } };
const pooledDs = await loadAll('mansion');
const areaSpec = { ...spec, pooled: false, ref: { ...spec.ref, areaStep: 5, stationStep: 5 } }, cityArea = fit(ds, all(ds.n), areaSpec, meta);
const pooled = fit(pooledDs, all(pooledDs.n), { ...spec, pooled: true }, meta, { cluster: true });
console.log(JSON.stringify({ city: dump(city), city_area: dump(cityArea), pooled: dump(pooled), predict: predict(city, prop),
  predict_area: [[72, 5], [500, 90]].map(([area, station_min]) => predict(cityArea, { ...prop, area, station_min })), predict_district: `${ds.wards[ds.districts[3].ward]} ${ds.districts[3].name}`,
  age_effect: ageEffect(city, 12, 31), age_effect_far: ageEffect(city, 10, 200) }));
