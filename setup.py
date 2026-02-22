"""
Custom setup.py needed because the package root (plantain2asr/) IS the git repo root.
pyproject.toml alone can't handle this layout without restructuring.
"""
from setuptools import setup, find_packages

# Sub-packages found inside the repo root
_subpackages = find_packages(".")

packages     = ["plantain2asr"] + [f"plantain2asr.{p}" for p in _subpackages]
package_dir  = {"plantain2asr": "."}
for sub in _subpackages:
    package_dir[f"plantain2asr.{sub}"] = sub.replace(".", "/")

setup(
    packages=packages,
    package_dir=package_dir,
)
