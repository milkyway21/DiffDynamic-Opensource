# Third-party source and resources

The repository retains the existing upstream license in `LICENSE`. Individual
third-party headers and data licenses continue to apply to their files.

- The network backbones and diffusion formulation derive from TargetDiff.
- `src/diffdynamic/chemistry/reconstruct_targetdiff_legacy.py` is a byte-frozen
  reconstruction implementation with its original liGAN attribution and GPLv2
  notice. Its SHA256 is checked by Prudent regression tests. Do not reformat it.
- SA scoring retains its RDKit Contrib attribution and fragment score resource.
- Lilly medchem rules retain their source notices.
- Optional `rdkit-structure-repair` remains a separate bundled library.
- Browser libraries and their license banners remain in the UI assets.

Pretrained model files are distributed separately using `weights-manifest.json`;
source export does not include their binary contents.
