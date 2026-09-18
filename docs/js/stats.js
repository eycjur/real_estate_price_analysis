// 集計まわりの小さな関数。numpy / pandas の既定の挙動に合わせてある。

/** 線形補間の分位点 (numpy.quantile の既定)。values は昇順でなくてよい。 */
export function quantile(values, q) {
  const v = Float64Array.from(values).sort();
  if (!v.length) return NaN;
  const pos = (v.length - 1) * q, lo = Math.floor(pos), hi = Math.ceil(pos);
  return v[lo] + (v[hi] - v[lo]) * (pos - lo);
}
export const median = values => quantile(values, 0.5);

/** 区分線形補間 (numpy.interp)。範囲外は端の値。 */
export function interp(x, xs, ys) {
  if (x <= xs[0]) return ys[0];
  const n = xs.length;
  if (x >= xs[n - 1]) return ys[n - 1];
  let i = 1;
  while (xs[i] < x) i++;
  return ys[i - 1] + (ys[i] - ys[i - 1]) * (x - xs[i - 1]) / (xs[i] - xs[i - 1]);
}

export function linspace(a, b, n) {
  return Array.from({ length: n }, (_, i) => a + (b - a) * i / (n - 1));
}

/** 相補誤差関数 (Abramowitz–Stegun 7.1.26、絶対誤差 1.5e-7)。p値の表示用。 */
export function erfc(x) {
  const z = Math.abs(x), t = 1 / (1 + 0.3275911 * z);
  const y = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))) * Math.exp(-z * z);
  return x >= 0 ? y : 2 - y;
}

/** 再現可能な乱数 (mulberry32)。標本抽出や検証用データの分割に使う。 */
export function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** items から最大 n 個を無作為に選ぶ(順序は保たない)。 */
export function sample(items, n, seed = 0) {
  if (items.length <= n) return Array.from(items);
  const a = Array.from(items), r = rng(seed);
  for (let i = 0; i < n; i++) {
    const j = i + Math.floor(r() * (a.length - i));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a.slice(0, n);
}
