from pathlib import Path
from typing import List, Optional
from .base import BaseASRDataset
from .types import AudioSample


class NeMoDataset(BaseASRDataset):
    """
    Dataset loader for the NeMo / JSONL manifest format.

    The manifest file (``manifest.jsonl``) must reside in ``rootdir_path`` and
    follow the NeMo convention — one JSON object per line with at minimum an
    ``audio_filepath`` field.  The ``text`` and ``duration`` fields are optional
    but strongly recommended for metric computation.

    Example manifest line::

        {"audio_filepath": "audio/sample.wav", "text": "hello world", "duration": 3.2}

    Args:
        rootdir_path: Path to the dataset root directory.
        name: Logical name used for cache directory naming.  Defaults to the
              directory name.
        limit: Cap the number of loaded samples (useful for quick tests).
        manifest_filename: Override the manifest file name (default: ``manifest.jsonl``).

    Example::

        from plantain2asr.dataloaders import NeMoDataset
        from plantain2asr.models import Models

        dataset = NeMoDataset("data/my_dataset")
        dataset.apply(Models.GigaAM_v3(device="cuda"), metrics_list=["wer", "cer"])
        df = dataset.to_pandas()
    """

    def __init__(
        self,
        rootdir_path: str,
        name: Optional[str] = None,
        limit: Optional[int] = None,
        manifest_filename: str = "manifest.jsonl",
    ):
        super().__init__()

        self.root_dir = Path(rootdir_path)
        self.name = name or self.root_dir.name
        self.limit = limit
        self.manifest_path = self.root_dir / manifest_filename

        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {self.manifest_path}\n"
                f"Expected a NeMo-format JSONL file at that path."
            )

        self._samples: List[AudioSample] = []
        self._load_manifest()
