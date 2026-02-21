from abc import ABC, abstractmethod
from typing import List, Union, Any, Tuple, Optional
from pathlib import Path
import time
import copy
from ..dataloaders.types import AudioSample

class BaseASRModel(ABC):
    """
    Абстрактный базовый класс для всех ASR моделей (локальных и облачных).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Уникальное имя модели (например, 'GigaAM-v3-CTC')"""
        pass

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
