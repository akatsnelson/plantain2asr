from typing import Optional
from .nemo_dataset import NeMoDataset


class DagrusDataset(NeMoDataset):
    """
    Convenience wrapper for the DaGRuS dataset (default name = "Dagrus").

    This is a thin subclass of :class:`NeMoDataset` kept for backwards
    compatibility.  New code should prefer :class:`NeMoDataset` directly.
    """

    def __init__(self, rootdir_path: str, limit: Optional[int] = None):
        super().__init__(rootdir_path=rootdir_path, name="Dagrus", limit=limit)
