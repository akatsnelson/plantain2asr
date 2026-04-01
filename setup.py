"""
Custom setup.py needed because the package root (plantain2asr/) IS the git repo root.
pyproject.toml alone can't handle this layout without restructuring.
"""
from setuptools import find_packages, setup
from setuptools.command.build_py import build_py as _build_py


class build_py(_build_py):
    """Exclude packaging helpers from the installed wheel."""

    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        if package != "plantain2asr":
            return modules
        return [module for module in modules if module[1] != "setup"]


# Sub-packages found inside the repo root
_subpackages = find_packages(".", exclude=("tests", "tests.*"))

packages = ["plantain2asr"] + [f"plantain2asr.{p}" for p in _subpackages]
package_dir = {"plantain2asr": "."}
for sub in _subpackages:
    package_dir[f"plantain2asr.{sub}"] = sub.replace(".", "/")

setup(
    packages=packages,
    package_dir=package_dir,
    cmdclass={"build_py": build_py},
)
