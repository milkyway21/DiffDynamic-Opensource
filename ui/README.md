# DiffDynamic browser interface

The existing HTML, Alpine.js, DOM IDs, and public JavaScript handler names are
preserved. There is no frontend build step or npm installation requirement.

## Workbench

`templates/index.html` loads ordered classic scripts from `static/js/workbench/`
for translation, account/API access, rendering helpers, generation, evaluation,
molecule browsing, history, configuration, pocket evaluation, optimization,
scaffold generation, cascade generation, job monitoring, uploads, and results.
The chemistry check and repair tab uses `workbench/medchem.js` and
`css/medchem.css`. The same options component is shared by all three generation
sources. Reports preserve original, primary, and candidate identities; SVG,
SDF, and docked PDBQT artifacts are fetched through authenticated report URLs.
`static/js/app.js` only binds the page after these features load. Keep scripts
classic: Alpine expressions and HTML event handlers use their global functions;
the shared lexical state also remains compatible with the original page.

The existing scaffold and molecule viewer scripts continue to load before the
workbench scripts. Do not load Alpine earlier than its current deferred script.

## Landing page

`templates/landing.html` contains page structure, its Three.js import map, and
static resource references. Its CSS lives in `static/css/landing.css`.
`static/js/landing/app.js` imports separate parsing, molecule rendering, protein
geometry, protein rendering, animation, hero/demo scene, diffusion-path/panel,
and logo modules. Diagnostic globals still load synchronously before the page
content, preserving the original error handlers and `log` function.

Showcase PDB/SDF content lives in `static/data/landing/`, loaded once before
building molecule scenes. Asset URLs use `import.meta.url` so `/zh/diffdynamic`
and `/en/diffdynamic` reverse-proxy prefixes work without rewriting JavaScript.
Include **all** these resources, the existing hero trajectories, Three.js,
Alpine.js, and 3Dmol.js when packaging a release.

## Verification

Run the dependency-free syntax, module linking, inline-handler, molecular-data,
and locale-path check with Node.js 18 or newer:

```sh
node --experimental-vm-modules ui/tests/check-contracts.cjs
```

`tests/test_refactor_frontend.py` runs the same check through pytest when Node is
available. The optional browser check requires Playwright and a matching browser:

```sh
node ui/tests/browser-smoke.cjs
```

It starts a temporary HTTP server on loopback, mocks API responses, tests the
root/Chinese/English routes at 375, 768, 1024, and 1440 px, and checks both pages
for JavaScript failures and missing assets. It never starts the real backend.
Chemistry workflow checks additionally cover mocked uploads, SMILES and history
inputs, docking prerequisites, original/candidate comparison, escaped findings,
cancellation, and the exact options sent by all three generation forms.
Set `DD_CHROMIUM_EXECUTABLE` for an existing compatible Chromium binary and
`DD_UI_SCREENSHOT_DIR` to save screenshots outside the release tree.
