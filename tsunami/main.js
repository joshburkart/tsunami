import init, { initThreadPool, run } from './pkg/tsunami.js';

var width = window.innerWidth
    || document.documentElement.clientWidth
    || document.body.clientWidth;
if (width < 700) {
    document.getElementById('canvas').style.display = 'none';
    document.getElementsByTagName('body')[0].style.padding = '20';
    document.getElementById('info').style.display = 'block';
} else {
    await init();
    await initThreadPool(navigator.hardwareConcurrency);
    await run();
}

