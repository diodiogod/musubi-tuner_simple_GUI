"""Fail before a compatibility test if the process loaded unexpected libraries.

Usage: python tools/verify_runtime_versions.py transformers=5.17.0 av=17.1.0
Use the same executable and environment as the following test command.
"""
import importlib
import sys


def main():
    for requirement in sys.argv[1:]:
        name, expected = requirement.split("=", 1)
        module = importlib.import_module(name)
        actual = str(module.__version__)
        print(f"{name}: {actual} ({module.__file__})")
        if actual != expected:
            raise SystemExit(f"Expected {name} {expected}; loaded {actual}. Tests must not proceed.")


if __name__ == "__main__":
    main()
