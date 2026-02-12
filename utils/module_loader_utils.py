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

    # Exclude directories that should not be imported automatically
    # past_diver: legacy code that conflicts with the current diver implementation
    # past_brainbert: legacy BrainBERT; use models/brainbert (reference-based) instead
    # past_popt: legacy PopT; use models/popt instead
    # Note: DIVER-1 is no longer excluded - its datasets module will fail to import
    # if lmdb is missing, but that's handled by try-except. DIVER-1's models and utils
    # modules are needed and will be imported correctly when needed.
    exclude_dirs = {'past_diver', 'past_brainbert', 'past_popt'}

    if recursive:
        # Recursively walk through all subpackages
        for _, module_name, is_pkg in pkgutil.walk_packages(package_path, prefix=f"{package_name}."):
            # Skip DIVER-1 and its submodules
            if any(excluded in module_name for excluded in exclude_dirs):
                continue
            try:
                importlib.import_module(module_name)
            except (ImportError, ModuleNotFoundError) as e:
                # Skip modules that fail to import (e.g., missing dependencies)
                # This allows the framework to continue even if some modules can't be imported
                pass
    else:
        # Only import direct children modules
        for _, module_name, _ in pkgutil.iter_modules(package_path):
            # Skip excluded directories
            if any(excluded in module_name for excluded in exclude_dirs):
                continue
            try:
                importlib.import_module(f"{package_name}.{module_name}")
            except (ImportError, ModuleNotFoundError) as e:
                # Skip modules that fail to import
                pass
