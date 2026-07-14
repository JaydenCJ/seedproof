"""Exception hierarchy for seedproof.

Every error raised on purpose by this package derives from
:class:`SeedproofError`, so callers (and the CLI) can catch one type and turn
it into a readable one-line message instead of a traceback.
"""

from __future__ import annotations


class SeedproofError(Exception):
    """Base class for all seedproof errors."""


class RecordError(SeedproofError):
    """A run record file is malformed, tampered with, or unsupported."""


class IngestError(SeedproofError):
    """A capture payload could not be converted into a run record."""


class AlignError(SeedproofError):
    """Two token streams cannot be compared on the requested basis."""


class MatrixError(SeedproofError):
    """A set of run records cannot be grouped into a matrix."""
