import init, { initThreadPool, run } from './pkg/tsunami.js';

await init();
await initThreadPool(navigator.hardwareConcurrency);
await run();
