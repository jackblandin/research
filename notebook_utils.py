import importlib
import sys


def import_module_by_name(mod_name):
    """
    Executes an "import" statement dynamically.

    Parameters
    ----------
    mod_name : str
        Name of the module to be imported.

    Returns
    -------
    module
        The imported module.
    """
    return importlib.__import__(mod_name)


def reload_module_by_name(mod_name, var_name):
    """
    Deletes and reimports a module (hot-module reloading).

    Parameters
    ----------
    mod_name : str
        Name of the module to be reloaded (e.g. 'numpy')
    var_name : str
        Name of the variable referencing the module (e.g. 'np')

    Returns
    -------
    """
    for mod in list(sys.modules.keys()):
        if mod_name in mod:
            del sys.modules[mod]
    if var_name in globals():
        del globals()[var_name]  # deletes the variable named <var_name>
    return importlib.__import__(mod_name)
