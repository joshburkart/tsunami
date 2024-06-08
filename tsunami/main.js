import init, { initThreadPool, run } from './pkg/tsunami.js';

await init();
await initThreadPool(1);
await run();
