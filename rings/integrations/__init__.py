from rings.integrations.study import SeparabilityStudy

__all__ = ["SeparabilityStudy", "SeparabilityCallback"]


def __getattr__(name):
    if name == "SeparabilityCallback":
        from rings.integrations.lightning import SeparabilityCallback

        return SeparabilityCallback
    raise AttributeError(f"module 'rings.integrations' has no attribute {name!r}")
