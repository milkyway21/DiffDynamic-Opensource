"""Install reusable code, historical imports, and the public Web/CLI resources.

Scientific dependencies remain managed by the existing environment. Building
this distribution never downloads dependencies or collects local run artifacts.
"""

import os
from pathlib import Path

from setuptools import find_namespace_packages, find_packages, setup
from setuptools.command.build import build as BaseBuild
from setuptools.command.build_py import build_py as BaseBuildPy
from setuptools.command.egg_info import egg_info as BaseEggInfo


ROOT = Path(__file__).resolve().parent
COMPATIBILITY_PACKAGES = ("datasets", "models", "utils", "server")
SCRIPT_MODULES = {
    "__init__",
    "sample_diffusion",
    "sample_for_pocket",
    "evaluate_diffusion",
    "check_model_type",
}
EXCLUDED_MODULES = {
    ("utils", "gspt1_class_setup"),
    ("utils", "gspt1_interface_setup"),
    ("utils", "ikzf2_scaffold_setup"),
}
LEGACY_MODULES = [
    "_diffdynamic_bootstrap",
    "batch_sampleandeval_parallel",
    "evaluate_pt_with_correct_reconstruct",
    "evaluate_pocket_quality",
    "extract_pt_to_sdf_excel",
    "scaffold_cascade_pipeline",
    "run_prudent_generations",
    "refine_saved_pt_baseline_then_eval",
    "split_and_eval_parallel",
    "show_sampling_steps",
]
RESOURCE_PATTERNS = (
    "configs/sampling.yml",
    "configs/benchmark_reference.yml",
    "configs/sampling_strategies/*.yml",
    "ui/templates/**/*",
    "ui/static/**/*",
    "ui/content/**/*",
    "server/server_config.example.json",
    "server/pt-loader-requirements.txt",
    "scripts/setup_pt_loader.sh",
    "start_server.sh",
    "README.md",
    "README.zh-CN.md",
    "weights-manifest.json",
    "rdkit-structure-repair/src/**/*.py",
    "rdkit-structure-repair/src/**/*.yaml",
    "rdkit-structure-repair/configs/*.yaml",
    "rdkit-structure-repair/configs/*.yml",
    "rdkit-structure-repair/README.md",
    "rdkit-structure-repair/LICENSE*",
)


def compatibility_packages():
    """Include old pickle namespaces, including former namespace subpackages."""
    packages = ["scripts", "paper_plot_scripts", "ui"]
    for name in COMPATIBILITY_PACKAGES:
        packages.append(name)
        packages.extend(
            name + "." + child
            for child in find_namespace_packages(
                str(ROOT / name),
                exclude=("__pycache__", "__pycache__.*", "tests", "tests.*"),
            )
            if "__pycache__" not in child.split(".")
        )
    return packages


class Build(BaseBuild):
    """Keep build artifacts in a cache unless a caller selects a temp path."""

    def initialize_options(self):
        super().initialize_options()
        self.build_base = os.environ.get(
            "DD_BUILD_BASE", str(ROOT / ".cache" / "package-build")
        )


class EggInfo(BaseEggInfo):
    """Avoid writing generated metadata into the reusable source tree."""

    def finalize_options(self):
        if self.egg_base is None:
            self.egg_base = os.environ.get(
                "DD_BUILD_METADATA_DIR", str(ROOT / ".cache" / "package-metadata")
            )
        Path(self.egg_base).mkdir(parents=True, exist_ok=True)
        super().finalize_options()


class BuildPy(BaseBuildPy):
    """Install a curated resource tree next to server and the legacy scripts."""

    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        return [
            item
            for item in modules
            if (package != "scripts" or item[1] in SCRIPT_MODULES)
            and (package, item[1]) not in EXCLUDED_MODULES
        ]

    def run(self):
        super().run()
        self._resource_outputs = []
        for pattern in RESOURCE_PATTERNS:
            for source in sorted(ROOT.glob(pattern)):
                if (
                    not source.is_file()
                    or source.is_symlink()
                    or "__pycache__" in source.parts
                    or any(
                        part.startswith(".") for part in source.relative_to(ROOT).parts
                    )
                ):
                    continue
                destination = Path(self.build_lib) / source.relative_to(ROOT)
                destination.parent.mkdir(parents=True, exist_ok=True)
                self.copy_file(str(source), str(destination))
                self._resource_outputs.append(str(destination))

    def get_outputs(self, include_bytecode=True):
        return super().get_outputs(include_bytecode) + getattr(
            self, "_resource_outputs", []
        )


setup(
    name="diffdynamic",
    version="0.2.0",
    description="Structure-based molecule generation and evaluation",
    packages=find_packages("src") + compatibility_packages(),
    package_dir={"diffdynamic": "src/diffdynamic"},
    py_modules=LEGACY_MODULES,
    package_data={
        "diffdynamic.evaluation": ["*.gz"],
        "server": ["server_config.example.json", "pt-loader-requirements.txt"],
    },
    include_package_data=False,
    python_requires=">=3.8",
    install_requires=[],
    entry_points={"console_scripts": ["diffdynamic=diffdynamic.cli.main:main"]},
    cmdclass={"build": Build, "build_py": BuildPy, "egg_info": EggInfo},
)
