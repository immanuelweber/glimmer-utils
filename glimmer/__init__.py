"""Top-level package for Glimmer Utils."""

try:
    from glimmer._version import __version__
except ImportError:
    # Fallback for development installs
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("glimmer")
    except PackageNotFoundError:
        __version__ = "0.0.0"

__all__ = ["__version__", "data", "lightning"]
