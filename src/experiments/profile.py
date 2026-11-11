"""公開 import パス。実装は ``src.experiments.execution.profile``。"""


def _export(module: object) -> list[str]:
    public: list[str] = []
    for name, value in vars(module).items():
        if name.startswith("__"):
            continue
        globals()[name] = value
        if not name.startswith("_"):
            public.append(name)
    return public


from src.experiments.execution import profile as _impl

__all__ = _export(_impl)
del _export, _impl
