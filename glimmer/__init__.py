"""Top-level package for Glimmer Utils."""

try:
    from ._version import __version__
except ImportError:
    # Fallback for development installs
    from importlib.metadata import version
    __version__ = version("glimmer")

__all__ = ["data", "lightning", "__version__"]
