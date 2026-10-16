/* Run with: node --experimental-vm-modules ui/tests/check-contracts.cjs */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { pathToFileURL, fileURLToPath } = require('node:url');

const root = path.resolve(__dirname, '../..');
const staticRoot = path.join(root, 'ui/static');
const read = (filename) => fs.readFileSync(filename, 'utf8');
const index = read(path.join(root, 'ui/templates/index.html'));
const landing = read(path.join(root, 'ui/templates/landing.html'));
const sources = (html) => [...html.matchAll(/(?:src|href)="(\/static\/[^"?]+)(?:\?[^\"]*)?"/g)]
    .map((match) => path.join(root, 'ui', match[1]));

function syntaxCheck(directory) {
    let count = 0;
    for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
        const filename = path.join(directory, entry.name);
        if (entry.isDirectory()) {
            count += syntaxCheck(filename);
        } else if (entry.name.endsWith('.js')) {
            const source = read(filename);
            if (/^\s*(?:import|export) /m.test(source)) {
                new vm.SourceTextModule(source, { identifier: filename });
            } else {
                new vm.Script(source, { filename });
            }
            count += 1;
        }
    }
    return count;
}

function checkWorkbenchGlobals() {
    const context = vm.createContext({
        localStorage: { getItem: () => null },
        document: {
            addEventListener() {},
            getElementById: () => ({}),
        },
        location: { pathname: '/en/diffdynamic/app' },
        DD_ROOT: '/en/diffdynamic',
        console,
    });
    context.window = context;
    for (const filename of sources(index).filter((filename) => filename.endsWith('.js'))) {
        if (filename.includes('/vendor/')) continue;
        vm.runInContext(read(filename), context, { filename });
    }
    assert.equal(vm.runInContext('API', context), '/en/diffdynamic');
    assert.equal(vm.runInContext('currentLang', context), 'en');
    assert.equal(vm.runInContext("t('gen.start')", context), 'Start Generation');
    assert.equal(vm.runInContext("t('mc.title')", context), 'Chemistry check & repair');
    assert.equal(vm.runInContext("medchemArtifactURL(42, 'pose/id')", context),
        '/en/diffdynamic/api/medchem/runs/42/artifacts/pose%2Fid');
    for (const name of ['initMedchem', 'buildMedchemRequest', 'generationMedchemOptions',
        'openMedchemReport', 'useRunForMedchem', 'cancelMedchemJob']) {
        assert.equal(vm.runInContext('typeof ' + name, context), 'function');
    }
    // DOM event attributes remain usable after splitting classic scripts.
    const ignored = new Set(['if', 'event', 'Number', 'String']);
    for (const handler of index.matchAll(/\bon\w+="([^"]+)"/g)) {
        for (const match of handler[1].matchAll(/(?:^|[^.\w])([A-Za-z_$][\w$]*)\s*\(/g)) {
            if (ignored.has(match[1])) continue;
            assert.equal(vm.runInContext(`typeof ${match[1]}`, context), 'function',
                'Missing inline handler: ' + match[1]);
        }
    }
}

async function checkLandingModules() {
    const context = vm.createContext({
        console,
        URL,
        fetch: async (url) => {
            const filename = fileURLToPath(url);
            return { ok: fs.existsSync(filename), status: 200, text: async () => read(filename) };
        },
    });
    const modules = new Map();
    function getModule(filename) {
        const normalized = path.resolve(filename);
        if (!modules.has(normalized)) {
            modules.set(normalized, new vm.SourceTextModule(read(normalized), {
                context,
                identifier: normalized,
                initializeImportMeta(meta) { meta.url = pathToFileURL(normalized).href; },
            }));
        }
        return modules.get(normalized);
    }
    function linker(specifier, referring) {
        let filename;
        if (specifier === 'three') {
            filename = path.join(staticRoot, 'libs/three/three.module.js');
        } else if (specifier.startsWith('three/addons/')) {
            filename = path.join(staticRoot, 'libs/three/addons', specifier.slice('three/addons/'.length));
        } else {
            assert.ok(specifier.startsWith('.'), 'Unexpected module dependency: ' + specifier);
            filename = path.resolve(path.dirname(referring.identifier), specifier);
        }
        return getModule(filename);
    }
    const app = getModule(path.join(staticRoot, 'js/landing/app.js'));
    await app.link(linker); // Verifies every module import/export and dependency exists.
    const data = getModule(path.join(staticRoot, 'js/landing/molecular-data.js'));
    const parser = getModule(path.join(staticRoot, 'js/landing/molecule-parser.js'));
    await data.evaluate();
    await parser.evaluate();
    await data.namespace.loadMolecularData();
    const protein = parser.namespace.parsePDB(data.namespace.PDB_DATA);
    const frames = parser.namespace.parseSDFTrajectory(data.namespace.SDF_TRAJECTORY);
    const ligand = parser.namespace.parseSDFModel(data.namespace.REF_LIGAND_SDF);
    assert.ok(protein.atoms.length > 1000, 'Showcase protein must remain intact');
    assert.ok(frames.length > 1, 'Showcase trajectory must retain multiple frames');
    assert.ok(ligand.atoms.length > 10, 'Showcase reference ligand must remain intact');
    const sampled = parser.namespace.resampleTrajectoryFrames(frames);
    assert.equal(sampled.length, 43);
    const positions = (frame) => frame.atoms.map(({ x, y, z, element }) => ({ x, y, z, element }));
    assert.deepEqual(positions(sampled[0]), positions(frames[0]));
    assert.deepEqual(positions(sampled[sampled.length - 1]), positions(frames[frames.length - 1]));
    // Relative assets follow either public locale prefix without HTML rewriting.
    for (const prefix of ['', '/zh/diffdynamic', '/en/diffdynamic']) {
        const base = `https://example.invalid${prefix}/static/js/landing/molecular-data.js`;
        assert.equal(new URL('../../data/landing/protein.pdb', base).pathname,
            `${prefix}/static/data/landing/protein.pdb`);
    }
    return modules.size;
}

(async () => {
    for (const filename of [...sources(index), ...sources(landing)]) {
        assert.ok(fs.existsSync(filename), 'Missing page asset: ' + filename);
    }
    assert.ok(!landing.includes('<style>'), 'Landing CSS must be a static asset');
    assert.ok(!landing.includes('const PDB_DATA'), 'Molecular data must stay outside HTML');
    assert.ok(index.indexOf('/workbench/jobs.js') < index.indexOf('/static/js/app.js'),
        'Feature declarations must precede startup');
    const scripts = syntaxCheck(staticRoot);
    checkWorkbenchGlobals();
    const modules = await checkLandingModules();
    console.log(`Frontend contracts passed: ${scripts} JavaScript files; ${modules} linked modules; locale paths and molecular assets.`);
})().catch((error) => {
    console.error(error);
    process.exitCode = 1;
});
