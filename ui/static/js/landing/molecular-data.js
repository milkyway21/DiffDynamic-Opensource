/** Bundled showcase structures; paths remain relative to this module. */
export let PDB_DATA = '';
export let SDF_TRAJECTORY = '';
export let REF_LIGAND_SDF = '';
let loading = null;

export function loadMolecularData() {
  if (!loading) {
    loading = Promise.all([
      'protein.pdb', 'trajectory.sdf', 'reference-ligand.sdf',
    ].map(async (filename) => {
      const url = new URL('../../data/landing/' + filename, import.meta.url);
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error('Unable to load showcase structure ' + filename +
          ': HTTP ' + response.status);
      }
      return response.text();
    })).then(([protein, trajectory, ligand]) => {
      PDB_DATA = protein;
      SDF_TRAJECTORY = trajectory;
      REF_LIGAND_SDF = ligand;
    });
  }
  return loading;
}
