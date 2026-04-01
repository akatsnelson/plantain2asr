from .base import BaseSection
from .metrics import MetricsSection
from .errors import ErrorFrequencySection
from .diff import DiffSection

__all__ = ["BaseSection", "MetricsSection", "ErrorFrequencySection", "DiffSection", "default_sections"]


def default_sections():
    """Canonical list of report sections, used by both ReportServer and ReportBuilder."""
    return [MetricsSection(), ErrorFrequencySection(), DiffSection()]
