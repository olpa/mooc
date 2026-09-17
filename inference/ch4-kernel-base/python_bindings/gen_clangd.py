"""Generate .clangd with absolute, machine-specific compile/link flags.

Run via `make clangd` from python_bindings/. Queries the project's venv
for torch's include/library dirs and the interpreter's own headers/libs,
since these vary by machine and can't be hardcoded or made relative.
"""
import sysconfig

from torch.utils.cpp_extension import include_paths, library_paths

lines = ["CompileFlags:", "  Add:"]

for path in include_paths():
    lines.append(f"    - -I{path}")
lines.append(f"    - -I{sysconfig.get_path('include')}")

for path in library_paths():
    lines.append(f"    - -L{path}")
python_libdir = sysconfig.get_config_var("LIBDIR")
if python_libdir:
    lines.append(f"    - -L{python_libdir}")

for lib in ("torch", "torch_cpu", "torch_python", "c10"):
    lines.append(f"    - -l{lib}")

print("\n".join(lines))
