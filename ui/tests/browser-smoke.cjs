/* Optional isolated browser check. Requires Playwright in NODE_PATH. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const http = require('node:http');
const path = require('node:path');
const { chromium } = require('playwright');
const { checkMedchem } = require('./medchem-browser.cjs');

const root = path.resolve(__dirname, '../..');
const output = process.env.DD_UI_SCREENSHOT_DIR;
const mime = { '.html': 'text/html', '.css': 'text/css', '.js': 'application/javascript', '.sdf': 'chemical/x-mdl-sdfile', '.pdb': 'chemical/x-pdb' };
const requests = [];
const misses = [];
const server = http.createServer((request, response) => {
    const url = new URL(request.url, 'http://localhost');
    const prefix = url.pathname.match(/^\/(?:zh|en)\/diffdynamic(?=\/|$)/)?.[0] || '';
    const pathname = url.pathname.slice(prefix.length) || '/';
    requests.push(url.pathname);
    if (pathname === '/favicon.ico') {
        response.writeHead(204);
        response.end();
        return;
    }
    if (pathname.startsWith('/api/')) {
        let body = {};
        if (pathname === '/api/me') body = { username: 'ui-test', is_admin: false };
        else if (pathname === '/api/jobs' || pathname === '/api/presets') body = [];
        response.writeHead(200, { 'Content-Type': 'application/json' });
        response.end(JSON.stringify(body));
        return;
    }
    let filename;
    if (pathname === '/' || pathname === '/app') {
        filename = path.join(root, 'ui/templates', pathname === '/' ? 'landing.html' : 'index.html');
    } else if (pathname.startsWith('/static/')) {
        filename = path.resolve(root, 'ui', '.' + pathname);
        if (!filename.startsWith(path.join(root, 'ui/static') + path.sep)) filename = null;
    }
    if (!filename || !fs.existsSync(filename)) {
        misses.push(url.pathname);
        response.writeHead(404);
        response.end();
        return;
    }
    let body = fs.readFileSync(filename);
    if (filename.endsWith('.html')) {
        body = body.toString().replace(/(["'])\/static\//g, `$1${prefix}/static/`)
            .replace(/href="\/app"/g, `href="${prefix}/app"`)
            .replace('__DD_PLATFORM_HOME__', '/');
        body = body.replace('</head>', `<script>window.DD_ROOT=${JSON.stringify(prefix)};window.DD_AUTH_MODE="off";</script></head>`);
    }
    response.writeHead(200, { 'Content-Type': mime[path.extname(filename)] || 'application/octet-stream' });
    response.end(body);
});

(async () => {
    await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
    const origin = `http://127.0.0.1:${server.address().port}`;
    const browser = await chromium.launch({ executablePath: process.env.DD_CHROMIUM_EXECUTABLE, headless: true, args: ['--use-angle=swiftshader', '--enable-webgl'] });
    try {
        for (const prefix of ['', '/zh/diffdynamic', '/en/diffdynamic']) {
            const context = await browser.newContext({ reducedMotion: 'reduce' });
            const page = await context.newPage();
            const errors = [];
            page.on('pageerror', (error) => errors.push(error.message));
            await page.goto(origin + prefix + '/app');
            await page.waitForFunction(() => typeof handleScaffold === 'function' && window.Alpine?.store('account'));
            await page.waitForFunction(() => document.getElementById('header-user-name').textContent === 'ui-test');
            assert.equal(await page.evaluate(() => API), prefix);
            await page.evaluate(() => switchToTab('generate'));
            for (const mode of ['custom', 'testset', 'batch']) {
                await page.selectOption('#gen-source', mode);
                assert.equal(await page.locator('#gen-section-' + mode).isVisible(), true);
            }
            await page.evaluate(() => switchToTab('scaffold'));
            await page.evaluate(() => selectOptFunc('scaffold'));
            await page.evaluate(() => toggleLang());
            assert.equal(await page.evaluate(() => typeof buildScaffoldRequest), 'function');
            await checkMedchem(page, prefix, output);
            for (const width of [375, 768, 1024, 1440]) {
                await page.setViewportSize({ width, height: 900 });
                if (output) {
                    fs.mkdirSync(output, { recursive: true });
                    await page.screenshot({ path: path.join(output, `workbench-${prefix.replaceAll('/', '_') || 'root'}-${width}.png`) });
                }
            }
            assert.deepEqual(errors, [], 'Workbench runtime errors');
            await context.close();
            const landingContext = await browser.newContext({ reducedMotion: 'reduce' });
            const landingPage = await landingContext.newPage();
            const landingErrors = [];
            landingPage.on('pageerror', (error) => landingErrors.push(error.message));
            await landingPage.goto(origin + prefix + '/');
            await landingPage.waitForFunction(() => document.getElementById('diag').textContent.includes('App initialization complete'), { timeout: 30000 });
            assert.ok((await landingPage.locator('#diag').textContent()).includes('Hero scene initialized'));
            assert.equal(await landingPage.locator('.hero-enter-btn').getAttribute('href'), prefix + '/app');
            for (const width of [375, 768, 1024, 1440]) {
                await landingPage.setViewportSize({ width, height: 900 });
                if (output) await landingPage.screenshot({ path: path.join(output, `landing-${prefix.replaceAll('/', '_') || 'root'}-${width}.png`) });
            }
            assert.deepEqual(landingErrors, [], 'Landing runtime errors');
            await landingContext.close();
            console.log('Browser smoke passed:', prefix || '/', 'at 375, 768, 1024, 1440px');
        }
        assert.deepEqual(misses, [], 'Missing frontend assets');
        for (const prefix of ['/zh/diffdynamic', '/en/diffdynamic']) {
            assert.ok(requests.includes(prefix + '/static/data/landing/protein.pdb'));
            assert.ok(requests.includes(prefix + '/static/data/hero-trajectories/004_trajectory_pymol.sdf'));
        }
    } finally {
        await browser.close();
    }
})().catch((error) => {
    console.error(error);
    process.exitCode = 1;
}).finally(() => server.close());
