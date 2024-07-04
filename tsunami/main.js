import init, { initThreadPool, run } from './pkg/tsunami.js';

Error.stackTraceLimit = 50;

await init();
await initThreadPool(navigator.hardwareConcurrency);
await run();
