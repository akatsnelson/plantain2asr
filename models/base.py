from abc import ABC, abstractmethod
from typing import List, Union, Any, Tuple, Optional, TYPE_CHECKING
from pathlib import Path
import time
import copy
from ..dataloaders.types import AudioSample
from ..core.processor import Processor

if TYPE_CHECKING:
    from ..dataloaders.base import BaseASRDataset


class BaseASRModel(Processor):
    """
    Абстрактный базовый класс для всех ASR моделей (локальных и облачных).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Уникальное имя модели (например, 'GigaAM-v3-CTC')"""
        pass

    @property
    def is_e2e(self) -> bool:
        """
        True, если модель умеет выводить пунктуацию (end-to-end).
        Такие модели следует оценивать с учётом пунктуации (do_clean=False).
        По умолчанию False — большинство моделей пунктуацию не выдают.
        """
        return False

    @abstractmethod
    def transcribe(self, audio_path: Union[str, Path]) -> str:
        """
        Распознает один аудиофайл (легаси API).
        """
        pass

    def transcribe_batch(self, audio_paths: List[Union[str, Path]]) -> List[str]:
        """
        Распознает список файлов (легаси API).
        """
        return [self.transcribe(p) for p in audio_paths]
    
    # ===== Новый унифицированный API =====
    
    def process_sample(self, sample: AudioSample, inplace: bool = False) -> AudioSample:
        """
        Распознает один AudioSample и заполняет его asr_results.
        """
        if not inplace:
            sample = copy.deepcopy(sample)
            
        start_time = time.time()
        try:
            hypothesis = self.transcribe(sample.audio_path)
            duration = time.time() - start_time
            error = None
        except Exception as e:
            duration = time.time() - start_time
            hypothesis = ""
            error = str(e)
            
        sample.add_result(self.name, hypothesis, duration, error)
        return sample
    
    def process_samples(self, samples: List[AudioSample], inplace: bool = False) -> List[AudioSample]:
        """
        Распознает список AudioSample (с поддержкой батчинга, если модель умеет).
        """
        if not inplace:
            samples = [copy.deepcopy(s) for s in samples]
            
        audio_paths = [s.audio_path for s in samples]
        
        try:
            # Пытаемся использовать батчинг
            batch_start = time.time()
            hypotheses = self.transcribe_batch(audio_paths)
            batch_duration = time.time() - batch_start
            
            if len(hypotheses) != len(samples):
                raise ValueError(
                    f"transcribe_batch returned {len(hypotheses)} hypotheses for {len(samples)} samples"
                )

            per_sample_duration = batch_duration / max(1, len(samples))

            for i, hyp in enumerate(hypotheses):
                samples[i].add_result(
                    self.name,
                    hyp,
                    per_sample_duration,
                    None,
                    batch_processing_time=batch_duration,
                )
                
        except Exception as e:
            print(f"⚠️ Batch processing failed for {self.name}, falling back to single sample: {e}")
            # Фоллбек на поштучную обработку
            for s in samples:
                self.process_sample(s, inplace=True)
                
        return samples

    # ===== Processor API =====

    def apply_to(self, dataset: 'BaseASRDataset') -> 'BaseASRDataset':
        """
        Интеграция с pipeline >>. Транскрибирует весь датасет.
        Параметры батчинга берутся из атрибутов модели (batch_size, save_step).
        """
        batch_size = getattr(self, 'batch_size', 32)
        save_step  = getattr(self, 'save_step', 32)
        dataset._apply_model(self, batch_size=batch_size, save_step=save_step)
        return dataset

    # ===== Training API =====

    def get_training_components(self) -> Tuple[Any, Any, Any]:
        """
        Возвращает компоненты для обучения модели.
        
        Returns:
            (model, processor, data_collator)
            - model: torch.nn.Module, который имеет метод forward(input_values, labels) -> loss
            - processor: Объект с методами __call__(audio) и tokenizer(text)
            - data_collator: Функция/класс для формирования батчей
            
        Raises:
            NotImplementedError: Если модель не поддерживает дообучение.
        """
        raise NotImplementedError(f"Model {self.name} does not support fine-tuning yet.")
