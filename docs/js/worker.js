// 計算用の Web Worker。画面からの { id, route, body } を service.js の関数に振り分ける。
import { routes } from './service.js';

self.onmessage = async ({ data: { id, route, body } }) => {
  try {
    // resources: Worker が読み込んだファイル(モジュールとデータ)の URL。画面の「キャッシュを消して再読み込み」で取り直す
    if (route === 'resources') return self.postMessage({ id, result: performance.getEntriesByType('resource').map(e => e.name) });
    if (!routes[route]) throw new Error(`不明な処理です: ${route}`);
    self.postMessage({ id, result: await routes[route](body ?? {}) });
  } catch (e) {
    self.postMessage({ id, error: e.message });
  }
};
