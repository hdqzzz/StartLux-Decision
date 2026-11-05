#!/usr/bin/env python3
"""Extract a pinned oMLX kernel subset without installing its inference server."""
import argparse
import hashlib
from pathlib import Path
import tempfile
import urllib.request
import zipfile

URL = "https://github.com/jundot/omlx/releases/download/v0.7.3/omlx-0.7.3-cp312-cp312-macosx_15_0_universal2.whl"
SHA256 = "ea314fd5f49e72a801a03f3775512a1bdf3e1bc5ba670f25f69cff27899efd40"
FILES = ["omlx/custom_kernels/_loader.py", "omlx/custom_kernels/quant_matmul/fast.py",
         "omlx/custom_kernels/quant_matmul/ane.py",
         "omlx/custom_kernels/quant_matmul/_ext.cpython-312-darwin.so",
         "omlx/custom_kernels/quant_matmul/omlx_quant_matmul_kernels.metallib",
         "omlx/custom_kernels/quant_matmul/omlx_quant_matmul_kernels_nax.metallib",
         "omlx/patches/qwen35_packed_linear.py"]
INITS = ["omlx", "omlx/custom_kernels", "omlx/custom_kernels/quant_matmul", "omlx/patches"]


def extract(wheel, destination):
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != SHA256:
        raise RuntimeError("oMLX wheel SHA256 mismatch")
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(wheel) as archive:
        for name in FILES:
            out = destination/name
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(archive.read(name))
        licenses = [x for x in archive.namelist() if x.endswith("/licenses/LICENSE")]
        if len(licenses) != 1:
            raise RuntimeError("expected exactly one upstream license")
        (destination/"OMLX-LICENSE").write_bytes(archive.read(licenses[0]))
    for name in INITS:
        # Do not import oMLX server bootstrap or mutate AGX environment variables.
        (destination/name/"__init__.py").write_text("")
    print("Prepared pinned oMLX0.7.3 kernel subset:", destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, help="reuse a locally downloaded, checksum-verified wheel")
    args = parser.parse_args()
    target = Path(__file__).resolve().parents[1]/"third_party/omlx-runtime"
    if args.wheel:
        extract(args.wheel, target)
    else:
        with tempfile.TemporaryDirectory(prefix="startlux-omlx-") as tmp:
            wheel = Path(tmp)/"omlx.whl"
            urllib.request.urlretrieve(URL, wheel)
            extract(wheel, target)


if __name__ == "__main__":
    main()
