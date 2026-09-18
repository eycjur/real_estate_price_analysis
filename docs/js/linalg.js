// 対称正定値行列まわりの最小限の線形代数。行列は行優先の Float64Array (k×k)。

/** X'X を先頭から順にコレスキー分解し、既出の列の線形結合になっている列を除いた列番号を返す。 */
export function independentColumns(A, k, tol = 1e-9) {
  const L = new Float64Array(k * k);  // L[j*k + i]: 列 j の、採用済み i 番目の列に対する成分
  const keep = [];
  const row = new Float64Array(k);
  for (let j = 0; j < k; j++) {
    for (let i = 0; i < keep.length; i++) {
      const c = keep[i];
      let s = A[j * k + c];
      for (let t = 0; t < i; t++) s -= row[t] * L[c * k + t];
      row[i] = s / L[c * k + i];
    }
    let d = A[j * k + j];
    for (let i = 0; i < keep.length; i++) d -= row[i] * row[i];
    if (d > tol * Math.max(A[j * k + j], 1e-300)) {
      for (let i = 0; i < keep.length; i++) L[j * k + i] = row[i];
      L[j * k + keep.length] = Math.sqrt(d);
      keep.push(j);
    }
  }
  return keep;
}

/** コレスキー分解 A = LL' (下三角 L を返す)。 */
export function cholesky(A, k) {
  const L = new Float64Array(k * k);
  for (let i = 0; i < k; i++) {
    for (let j = 0; j <= i; j++) {
      let s = A[i * k + j];
      for (let t = 0; t < j; t++) s -= L[i * k + t] * L[j * k + t];
      if (i === j) {
        if (!(s > 0)) throw new Error('行列が正定値ではありません');
        L[i * k + i] = Math.sqrt(s);
      } else L[i * k + j] = s / L[j * k + j];
    }
  }
  return L;
}

/** LL'x = b を解く。 */
export function cholSolve(L, k, b) {
  const x = Float64Array.from(b);
  for (let i = 0; i < k; i++) {
    let s = x[i];
    for (let t = 0; t < i; t++) s -= L[i * k + t] * x[t];
    x[i] = s / L[i * k + i];
  }
  for (let i = k - 1; i >= 0; i--) {
    let s = x[i];
    for (let t = i + 1; t < k; t++) s -= L[t * k + i] * x[t];
    x[i] = s / L[i * k + i];
  }
  return x;
}

/** (LL')⁻¹ */
export function cholInverse(L, k) {
  const inv = new Float64Array(k * k), e = new Float64Array(k);
  for (let j = 0; j < k; j++) {
    e.fill(0); e[j] = 1;
    const x = cholSolve(L, k, e);
    for (let i = 0; i < k; i++) inv[i * k + j] = x[i];
  }
  return inv;
}

/** k×k の積 A·B */
export function matmul(A, B, k) {
  const C = new Float64Array(k * k);
  for (let i = 0; i < k; i++) {
    for (let t = 0; t < k; t++) {
      const a = A[i * k + t];
      if (a === 0) continue;
      for (let j = 0; j < k; j++) C[i * k + j] += a * B[t * k + j];
    }
  }
  return C;
}
