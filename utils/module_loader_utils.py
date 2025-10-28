import importlib
import os
import pkgutil

def import_all_from_package(package_name, recursive=False):
    """
    Import all modules from a package.

    Args:
        package_name: Name of the package to import modules from
        recursive: If True, recursively import all subpackages and their modules
    """
    package = importlib.import_module(package_name)
    package_path = package.__path__

    if recursive:
        # Recursively walk through all subpackages
        for _, module_name, is_pkg in pkgutil.walk_packages(package_path, prefix=f"{package_name}."):
            importlib.import_module(module_name)
    else:
        # Only import direct children modules
        for _, module_name, _ in pkgutil.iter_modules(package_path):
            importlib.import_module(f"{package_name}.{module_name}")
