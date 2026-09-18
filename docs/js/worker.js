// 計算用の Web Worker。画面からの { id, route, body } を service.js の関数に振り分ける。
import { routes } from './service.js';

self.onmessage = async ({ data: { id, route, body } }) => {
  try {
    if (!routes[route]) throw new Error(`不明な処理です: ${route}`);
    self.postMessage({ id, result: await routes[route](body ?? {}) });
  } catch (e) {
    self.postMessage({ id, error: e.message });
  }
};
