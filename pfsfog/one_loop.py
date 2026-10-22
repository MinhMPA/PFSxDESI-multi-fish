"""Select and authenticate the ELF one-loop source used by new forecasts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

ELF_ROOT = Path("/Users/nguyenmn/elf")
SOURCE_MANIFEST = Path(__file__).with_name("elf_source.json")


def reject_legacy_ap_qualification(backend):
    """The ps AP-table certificates do not measure ELF's native remapping."""
    if getattr(backend, "native_full_mean_ap", False):
        raise ValueError("ELF requires new physical evidence from scripts/validate_elf_engine.py; "
                         "the old ps AP-table qualification is inapplicable")


def elf_source_identity():
    """Verify the imported package, complete pinned source inventory and bytes.

    The local path is intentional: an unrelated installed package named elf
    must not silently become the scientific calculation selected by the user.
    """
    import elf
    root = ELF_ROOT.resolve()
    package = root/"src/elf"
    if Path(elf.__file__).resolve() != package/"__init__.py":
        raise ValueError("ELF import origin differs from the selected local source")
    for name, module in list(sys.modules.items()):
        if name == "elf" or name.startswith("elf."):
            origin = getattr(module, "__file__", None)
            if origin and not Path(origin).resolve().is_relative_to(package):
                raise ValueError("ELF module origin differs: "+name)
    pinned = json.loads(SOURCE_MANIFEST.read_text())
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                          capture_output=True, text=True).stdout.strip()
    if head != pinned["head"]:
        raise ValueError("ELF source revision differs; qualify and pin the new revision explicitly")
    actual_inventory = {str(p.relative_to(root)) for p in package.rglob("*")
                        if p.is_file() and p.suffix in (".py", ".txt")}
    expected_inventory = {p for p in pinned["files"] if p.startswith("src/elf/")}
    if actual_inventory != expected_inventory:
        raise ValueError("ELF source inventory differs")
    files = {}
    for name, expected in pinned["files"].items():
        path = root/name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError("ELF source bytes differ: "+str(path))
        files[str(path)] = digest
    return {"root": str(root), "head": head, "module": str(package/"__init__.py"),
            "file_hashes": files,
            "manifest_sha256": hashlib.sha256(SOURCE_MANIFEST.read_bytes()).hexdigest()}


def make_one_loop_engine(**settings):
    """Construct the selected Eulerian, IR-off, renormalized ELF calculation."""
    elf_source_identity()
    from .elf_adapter import ELFPowerSpectrum
    return ELFPowerSpectrum(**settings)
