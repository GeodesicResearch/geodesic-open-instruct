# Re-export everything so `from open_instruct.utils import X` still works.
#
# Uses lazy loading via __getattr__ to avoid importing modules with heavy deps
# (ray, torch, vllm, deepspeed) at package init time. The code execution server
# (Singularity container) imports open_instruct.utils.logger, which triggers
# this __init__.py — eager imports would fail because those deps are not
# installed in the container.
import importlib as _importlib

# Submodules that can be accessed as `from open_instruct.utils import logger` etc.
_SUBMODULES = {
    "beaker",
    "checkpoints",
    "cli",
    "datasets",
    "deepspeed",
    "flops",
    "general",
    "ground_truth",
    "grpo",
    "judge",
    "launch",
    "logger",
    "math",
    "model",
    "rl",
    "ray",
    "ulysses",
    "vllm",
}

# Modules whose `*` exports are re-exported from this package for backwards
# compatibility (e.g. `from open_instruct import utils; utils.ModelDims`).
_STAR_REEXPORT_MODULES = [
    "beaker",
    "checkpoints",
    "cli",
    "datasets",
    "deepspeed",
    "flops",
    "general",
    "ray",
    "ulysses",
]

# Cache for loaded modules and their exported names
_loaded_stars: dict[str, dict] = {}


def _load_star_exports(module_name: str) -> dict:
    """Import a submodule and return its public names (respecting __all__)."""
    if module_name not in _loaded_stars:
        mod = _importlib.import_module(f"open_instruct.utils.{module_name}")
        if hasattr(mod, "__all__"):
            names = {name: getattr(mod, name) for name in mod.__all__}
        else:
            names = {name: getattr(mod, name) for name in dir(mod) if not name.startswith("_")}
        _loaded_stars[module_name] = names
    return _loaded_stars[module_name]


def __getattr__(name: str):
    # First check if it's a submodule name
    if name in _SUBMODULES:
        return _importlib.import_module(f"open_instruct.utils.{name}")
    # Then check star-reexported names
    for mod_name in _STAR_REEXPORT_MODULES:
        try:
            exports = _load_star_exports(mod_name)
            if name in exports:
                return exports[name]
        except ImportError:
            continue
    raise AttributeError(f"module 'open_instruct.utils' has no attribute {name!r}")
