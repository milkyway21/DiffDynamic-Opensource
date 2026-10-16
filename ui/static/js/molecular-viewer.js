/* Shared protein ribbon and ligand display for generation and scaffold views. */
window.MolecularViewer = (() => {
  const proteinColors = { h: '#79b5dc', s: '#aed6ee', c: '#79b5dc' };

  function create(containerId) {
    let viewer;
    try {
      const probe = document.createElement('canvas');
      const gl = probe.getContext('webgl') || probe.getContext('experimental-webgl');
      if (gl && window.$3Dmol) {
        const lose = gl.getExtension('WEBGL_lose_context');
        if (lose) lose.loseContext();
        viewer = $3Dmol.createViewer(containerId, { backgroundColor: '#0a1628' });
      }
    } catch (_) { /* Use the interactive CPU view if WebGL is unavailable. */ }
    return viewer || createScaffoldCanvasViewer(containerId);
  }

  function clear(viewer) {
    viewer.removeAllModels();
    viewer.removeAllShapes();
    viewer.removeAllLabels();
  }

  function load(viewer, context, onPick = null) {
    clear(viewer);
    const protein = viewer.addModel(context.protein_pdb, 'pdb');
    protein.setStyle({}, { cartoon: {
      colorscheme: { prop: 'ss', map: proteinColors },
      opacity: 0.9, thickness: 0.35,
    } });
    let ligand = null;
    if (context.ligand_sdf) {
      ligand = viewer.addModel(context.ligand_sdf, 'sdf');
      ligand.setStyle({}, {
        stick: { radius: 0.16, color: '#a9b8c7' },
        sphere: { scale: 0.2, color: '#a9b8c7' },
      });
      if (context.scaffold_indices && context.scaffold_indices.length) {
        ligand.setStyle({ index: context.scaffold_indices }, {
          stick: { radius: 0.22, color: '#5ec7a4' },
          sphere: { scale: 0.23, color: '#5ec7a4' },
        });
      }
      if (onPick) ligand.setClickable({}, true, atom => onPick(atom, true));
    }
    if (onPick) protein.setClickable({}, true, atom => onPick(atom, false));
    const fit = () => {
      if (viewer.getView && viewer.setView) {
        const view = viewer.getView();
        view.splice(4, 4, 0, 0, 0, 1);
        viewer.setView(view);
      } else if (viewer.resetOrientation) viewer.resetOrientation();
      viewer.zoomTo({ model: (ligand || protein).getID() });
      viewer.zoom(0.8);
      viewer.render();
    };
    fit();
    return { protein, ligand, fit };
  }

  return { create, load, clear };
})();
