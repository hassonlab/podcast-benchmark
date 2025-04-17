import importlib
import os
import pkgutil

def import_all_from_package(package_name):
    package = importlib.import_module(package_name)
    package_path = package.__path__

    for _, module_name, _ in pkgutil.iter_modules(package_path):
        importlib.import_module(f"{package_name}.{module_name}")
