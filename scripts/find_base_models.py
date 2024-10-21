"""Find all pydantic BaseModel classes in a package, including indirect inheritance."""

import contextlib
import importlib
import inspect
import pkgutil
import types

import typer
from pydantic import BaseModel


def find_subclasses_in_module(
    module: types.ModuleType,
) -> list[tuple[str, str, list[str]]]:
    subclasses: list[tuple[str, str, list[str]]] = []

    for name, obj in inspect.getmembers(module, inspect.isclass):
        # Ensure the class is defined in the current module to avoid duplicates from
        # import
        if obj.__module__ != module.__name__:
            continue
        # Instantiation of generic class
        if "[" in name:
            continue
        if issubclass(obj, BaseModel) and obj is not BaseModel:
            # Get immediate base classes excluding 'object' and 'BaseModel'
            base_names = [
                base.__name__
                for base in obj.__bases__
                if base not in (object, BaseModel)
            ]
            subclasses.append((module.__name__, name, base_names))

    return subclasses


def import_submodules(package_name: str) -> list[types.ModuleType]:
    """Recursively import all submodules of a package."""
    package = importlib.import_module(package_name)
    submodules = [package]

    for _, name, _ in pkgutil.walk_packages(package.__path__, package_name + "."):
        with contextlib.suppress(Exception):
            submodules.append(importlib.import_module(name))

    return submodules


app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
)


@app.command(help=__doc__)
def main(package_name: str) -> None:
    prefix = f"{package_name}."
    classes_unique: set[str] = set()
    classes_output: list[str] = []

    for module in import_submodules(package_name):
        try:
            subclasses = find_subclasses_in_module(module)
        except Exception:
            continue

        for module_name, class_name, base_names in subclasses:
            class_id = f"{module_name}.{class_name}"
            if class_id in classes_unique:
                continue
            classes_unique.add(class_id)

            bases_str = f"({', '.join(base_names)})" if base_names else ""
            classes_output.append(
                f"{module_name.removeprefix(prefix)}::{class_name}{bases_str}"
            )

    print("\n".join(classes_output))


if __name__ == "__main__":
    app()
