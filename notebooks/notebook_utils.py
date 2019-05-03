import importlib
import sys


def import_module_by_name(mod_name):
    return importlib.__import__(mod_name)


def reload_module_by_name(mod_name, var_name):
    for mod in list(sys.modules.keys()):
        if mod_name in mod:
            del sys.modules[mod]
    if var_name in globals():
        del globals()[var_name] # deletes the variable named <var_name>
    return importlib.__import__(mod_name)
