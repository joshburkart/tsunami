import { defineConfig } from "vite";

const viteHeaderPlugin = {
    name: 'add headers',
    configureServer: (server) => {
        server.middlewares.use((req, res, next) => {
            res.setHeader("Access-Control-Allow-Origin", "*");
            res.setHeader("Access-Control-Allow-Methods", "GET");
            res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
            res.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
            next(); dddd
        });
    }
};

export default defineConfig({
    build: {
        target: 'esnext'
    },
    plugins: [viteHeaderPlugin],
});
