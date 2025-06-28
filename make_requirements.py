#!/usr/bin/env python3
import os
import ast
import sys
from importlib import metadata

# --- CONFIGURE THIS ---
PROJECT_PATH = r"C:\Users\santi\Desktop\Oto\Vertebra-Landmark-Detection"
OUTPUT_FILE  = os.path.join(PROJECT_PATH, "requirements.txt")
# ----------------------

def find_py_files(root):
    for dirpath, dirnames, filenames in os.walk(root):
        # skip __pycache__
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for fname in filenames:
            if fname.endswith(".py"):
                yield os.path.join(dirpath, fname)

def collect_imports(py_path):
    with open(py_path, "r", encoding="utf8") as f:
        node = ast.parse(f.read(), filename=py_path)
    imports = set()
    for stmt in ast.walk(node):
        if isinstance(stmt, ast.Import):
            for n in stmt.names:
                imports.add(n.name.split(".")[0])
        elif isinstance(stmt, ast.ImportFrom):
            if stmt.module and stmt.level == 0:
                imports.add(stmt.module.split(".")[0])
    return imports

def is_local_module(mod_name, project_root):
    # if there's a folder or file matching mod_name in project, treat as local
    path1 = os.path.join(project_root, mod_name + ".py")
    path2 = os.path.join(project_root, mod_name)
    return os.path.exists(path1) or os.path.exists(path2)

def main():
    all_imports = set()
    for py in find_py_files(PROJECT_PATH):
        all_imports |= collect_imports(py)

    # filter out builtins, stdlib, and local modules
    externals = set()
    for mod in sorted(all_imports):
        if is_local_module(mod, PROJECT_PATH):
            continue
        try:
            # try to see if it's installed as a distribution
            dist = metadata.distribution(mod)
            externals.add(f"{dist.metadata['Name']}=={dist.version}")
        except metadata.PackageNotFoundError:
            # not a top-level distribution, maybe stdlib or nested import
            # skip modules that come with the stdlib
            # crude check: if we can import and it's in stdlib path, skip
            try:
                m = __import__(mod)
                if hasattr(m, "__file__") and "site-packages" in (m.__file__ or ""):
                    # lives in site-packages but dist metadata missing: include without version
                    externals.add(mod)
            except ImportError:
                pass

    # write requirements.txt
    with open(OUTPUT_FILE, "w", encoding="utf8") as out:
        for line in sorted(externals):
            out.write(line + "\n")

    print(f"Written {len(externals)} packages to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
