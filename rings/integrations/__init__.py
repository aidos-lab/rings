from rings.integrations.study import SeparabilityStudy

__all__ = [
    "SeparabilityStudy",
    "SeparabilityCallback",
    "DGLOriginal",
    "DGLEmptyFeatures",
    "DGLRandomFeatures",
    "DGLEmptyGraph",
    "DGLCompleteGraph",
    "DGLRandomGraph",
]


def __getattr__(name):
    if name == "SeparabilityCallback":
        from rings.integrations.lightning import SeparabilityCallback

        return SeparabilityCallback

    if name in [
        "DGLOriginal",
        "DGLEmptyFeatures",
        "DGLRandomFeatures",
        "DGLEmptyGraph",
        "DGLCompleteGraph",
        "DGLRandomGraph",
    ]:
        from rings.integrations import dgl

        return getattr(dgl, name)

    raise AttributeError(f"module 'rings.integrations' has no attribute {name!r}")
