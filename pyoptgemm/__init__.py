# Read version number as written by setuptools_scm
# try:
#   from pyoptgemm.version import version as __version__  # @manual
# except Exception:  # pragma: no cover
#     __version__ = "Unknown"

from .optgemm import mm

__all__ = [mm]