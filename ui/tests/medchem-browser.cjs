/* Workflow assertions used by browser-smoke.cjs; all data are synthetic. */
const assert = require('node:assert/strict');

const variant = (stage, id) => ({
    variant_id: id, stage, smiles: 'CCO',
    properties: { qed: .41, sa: .8, mw: 46.07, logp: -.01, tpsa: 20.23 },
    findings: [], changes: [{ rule_id: 'fixture', message: 'Synthetic change' }],
    geometry: { status: 'passed', coordinate_source: 'generated_conformer', pocket_status: 'not_applicable' },
    evaluation: { status: 'skipped', reason: 'No docking requested' },
    depiction_artifact_id: id + '-svg', structure_artifact_id: id + '-sdf',
    structure_format: 'sdf', atom_mapping: { 0: 0, 1: 1 },
});
const report = {
    schema_version: 1, run_id: 91, status: 'partial',
    summary: { total: 2, processed: 2, valid: 1, repaired: 1, candidates: 2, failed: 1 },
    records: [
        { record_id: 'record-1', name: 'Synthetic alcohol', status: 'completed',
            original: variant('original', 'original'), primary: {
                ...variant('primary', 'primary'),
                main_evaluation: { status: 'completed', mode: 'score_only', affinity: -4.1 },
            },
            candidates: [variant('candidate', 'candidate-1'),
                { ...variant('candidate', 'candidate-2'),
                    protein_artifact_id: 'protein-2',
                    evaluation: { status: 'completed', affinity: -5.2, pose_artifact_id: 'pose-2' } }],
            findings: [{ message: '<img src=x onerror="window.mcUnsafe=true">' }], changes: [] },
        { record_id: 'record-2', name: 'Invalid record', status: 'failed', error: 'Could not parse record',
            original: null, primary: null, candidates: [], findings: [], changes: [] },
    ],
    pockets: [{ label: 'Pocket A', status: 'completed', protein_artifact_id: 'protein-1' },
        { label: 'Pocket B', status: 'completed', protein_artifact_id: 'protein-2' }],
    artifacts: [{ artifact_id: 'summary', format: 'csv', label: 'CSV' },
        { artifact_id: 'protein-1', format: 'pdb', label: 'Pocket A' },
        { artifact_id: 'protein-2', format: 'pdb', label: 'Pocket B' }],
};

async function checkMedchem(page, prefix, output) {
    const bodies = [];
    const generationBodies = [];
    let jobStatus = 'completed';
    await page.route('**/api/medchem', async route => {
        bodies.push(route.request().postDataJSON());
        await route.fulfill({ json: { job_id: 'medchem_fixture', run_id: 91, status: 'running' } });
    });
    await page.route('**/api/jobs/medchem_fixture', async route => {
        if (route.request().method() === 'DELETE') jobStatus = 'cancelled';
        await route.fulfill({ json: { job_id: 'medchem_fixture', run_id: 91, status: jobStatus, progress: 1 } });
    });
    await page.route('**/api/medchem/runs/*/report', route => route.fulfill({ json: report }));
    await page.route('**/api/medchem/runs/*/artifacts/*', route => {
        if (route.request().url().includes('/protein-')) return route.fulfill({
            contentType: 'chemical/x-pdb',
            body: 'ATOM      1  CA  ALA A   1      20.000   0.000   0.000  1.00  0.00           C\nEND\n',
        });
        if (route.request().url().endsWith('/pose-2')) return route.fulfill({
            contentType: 'chemical/x-pdbqt',
            body: 'ROOT\nATOM      1  C   UNL     1       0.000   0.000   0.000  1.00  0.00     0.000 C\nENDROOT\nTORSDOF 0\n',
        });
        return route.fulfill({ contentType: 'image/svg+xml', body: '<svg xmlns="http://www.w3.org/2000/svg" width="200" height="150"><text x="60" y="75">C—C—O</text></svg>' });
    });
    await page.route('**/api/upload', route => route.fulfill({ json: {
        stored_path: '/uploads/fixture.sdf', filename: 'fixture.sdf', size: 8,
    } }));
    for (const endpoint of ['/api/generate', '/api/generate/custom', '/api/generate/batch']) {
        await page.route('**' + endpoint, async route => {
            generationBodies.push({ endpoint, body: route.request().postDataJSON() });
            await route.fulfill({ json: { job_id: 'generation_fixture', run_id: 101, status: 'running' } });
        });
    }
    await page.evaluate(() => switchToTab('medchem'));
    const chooserEvent = page.waitForEvent('filechooser');
    await page.locator('#mc-input-file button').last().click();
    await (await chooserEvent).setFiles({ name: 'fixture.sdf', mimeType: 'chemical/x-mdl-sdfile', buffer: Buffer.from('fixture\n') });
    await page.waitForFunction(() => document.getElementById('mc-input-path').value === '/uploads/fixture.sdf');
    assert.equal((await page.evaluate(() => buildMedchemRequest())).input_path, '/uploads/fixture.sdf');
    await page.selectOption('#mc-source', 'smiles');
    await page.fill('#mc-smiles', 'CCO\ninvalid');
    assert.equal(await page.locator('#mc-input-file').isVisible(), false);
    assert.equal(await page.locator('#mc-aligned').isDisabled(), true);
    const request = await page.evaluate(() => buildMedchemRequest());
    assert.equal(request.smiles, 'CCO\ninvalid');
    assert.equal(request.input_aligned, false);
    assert.equal(request.options.operation, 'propose');
    assert.equal(request.options.evaluation.mode, 'chemistry_3d');
    await page.fill('#mc-smiles', Array(101).fill('CCO').join('\n'));
    assert.equal(await page.evaluate(() => {
        try { buildMedchemRequest(); return false; } catch (_) { return true; }
    }), true);
    await page.fill('#mc-smiles', 'CCO\ninvalid');
    await page.locator('input[name="mc-evaluation"][value="vina_dock"]').check();
    assert.equal(await page.locator('#mc-docking-options').isVisible(), true);
    const invalidDock = await page.evaluate(() => {
        try { buildMedchemRequest(); return false; } catch (_) { return true; }
    });
    assert.equal(invalidDock, true, 'Docking requires a protein and a defined box');
    await page.fill('#mc-protein', '/uploads/fixture.pdb');
    await page.selectOption('#mc-box-mode', 'explicit');
    for (const axis of ['x', 'y', 'z']) await page.fill('#mc-center-' + axis, '0');
    const dock = await page.evaluate(() => buildMedchemRequest());
    assert.deepEqual(dock.options.evaluation.center_xyz, [0, 0, 0]);
    assert.deepEqual(dock.options.evaluation.size_xyz, [20, 20, 20]);
    assert.equal(dock.options.evaluation.timeout_seconds, 120);
    await page.locator('input[name="mc-evaluation"][value="chemistry_3d"]').check();
    await page.click('#mc-submit');
    await page.waitForFunction(() => document.querySelectorAll('.mc-variant').length === 3);
    assert.equal(bodies.length, 1);
    assert.equal(bodies[0].smiles, 'CCO\ninvalid');
    assert.equal(await page.evaluate(() => Boolean(window.mcUnsafe)), false);
    assert.equal(await page.locator('#mc-report img[src="x"]').count(), 0);
    assert.ok((await page.locator('.mc-variant[data-stage="primary"]').textContent()).includes('-4.10 kcal/mol'));
    await page.selectOption('#mc-candidate-select', '1');
    assert.ok((await page.locator('.mc-variant[data-stage="candidate"]').textContent()).includes('-5.20 kcal/mol'));
    assert.equal(await page.locator('[data-mc-view-format="pdbqt"]').count(), 1);
    await page.locator('[data-mc-view-format="pdbqt"]').click();
    await page.waitForFunction(() => medchemState.viewers.length > 0);
    assert.equal(await page.evaluate(() => medchemState.viewers[0].getModel().selectedAtoms({}).length), 1);
    assert.equal(await page.evaluate(() => medchemState.viewers[0].getModel(0).selectedAtoms({})[0].x), 20);
    const downloadEvent = page.waitForEvent('download');
    await page.locator('[data-mc-download="pose-2"]').click();
    assert.equal((await downloadEvent).suggestedFilename(), 'run_91_pose-2.pdbqt');
    assert.equal(await page.evaluate(() => medchemArtifactURL(91, 'pose/id')), prefix + '/api/medchem/runs/91/artifacts/pose%2Fid');
    await page.selectOption('#mc-record-select', '1');
    assert.ok((await page.locator('#mc-report').textContent()).includes('Could not parse record'));
    await page.selectOption('#mc-record-select', '0');
    for (const width of [375, 768, 1024, 1440]) {
        await page.setViewportSize({ width, height: 1000 });
        await page.evaluate(() => scrollTo(0, 0));
        const overflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1);
        assert.equal(overflow, false, `Chemistry page overflows at ${width}px`);
        if (output) {
            const path = require('node:path');
            require('node:fs').mkdirSync(output, { recursive: true });
            await page.screenshot({ path: path.join(output, `medchem-${prefix.replaceAll('/', '_') || 'root'}-${width}.png`), fullPage: true });
        }
    }
    jobStatus = 'running';
    await page.click('#mc-submit');
    await page.waitForFunction(() => medchemState.busy);
    await page.click('#mc-cancel');
    await page.waitForFunction(() => !medchemState.busy);
    assert.equal(jobStatus, 'cancelled');
    await page.evaluate(() => useRunForMedchem(7));
    const historical = await page.evaluate(() => buildMedchemRequest());
    assert.equal(historical.source_run_id, 7);
    assert.equal(historical.smiles, undefined);
    await page.evaluate(() => switchToTab('generate'));
    assert.equal(await page.evaluate(() => generationMedchemOptions()), null);
    await page.check('#gen-medchem-enable');
    for (const source of ['custom', 'testset', 'batch']) {
        await page.selectOption('#gen-source', source);
        const options = await page.evaluate(() => generationMedchemOptions());
        assert.equal(options.operation, 'propose');
        assert.equal(options.profile, 'local');
    }
    await page.locator('input[name="gen-mc-profile"][value="aggressive"]').check();
    assert.equal(await page.evaluate(() => generationMedchemOptions().profile), 'aggressive');
    await page.selectOption('#gen-source', 'custom');
    await page.fill('#custom-protein', '/uploads/fixture.pdb');
    for (const [source, handler] of [['custom', 'handleCustomGenerate'], ['testset', 'handleGenerate'], ['batch', 'handleBatchGenerate']]) {
        await page.selectOption('#gen-source', source);
        await page.evaluate(name => window[name](), handler);
    }
    assert.equal(generationBodies.length, 3);
    assert.ok(generationBodies.every(({ body }) => body.medchem.profile === 'aggressive' && body.medchem.operation === 'propose'));
    await page.uncheck('#gen-medchem-enable');
    assert.equal(await page.evaluate(() => generationMedchemOptions()), null);
    await page.evaluate(() => handleBatchGenerate());
    assert.equal(generationBodies[3].body.medchem, null);
    await page.evaluate(() => stopPolling());
    await page.evaluate(() => switchToTab('scaffold'));
}

module.exports = { checkMedchem };
