from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import sys

@dataclass
class AudioSample:
    """
    Универсальный контейнер для аудио-сэмпла.
    Хранит входные данные и историю результатов транскрипции (asr_results).
    """
    # === Входные данные ===
    id: str                  # Уникальный ID
    audio_path: str          # Абсолютный путь к аудиофайлу
    
    # Опциональные входные данные
    text: Optional[str] = None       # Эталонный текст (reference)
    duration: float = 0.0            # Длительность аудио в секундах
    meta: Dict[str, Any] = field(default_factory=dict) 
    
    # === История результатов (для сравнения моделей) ===
    # Ключ: имя модели, Значение: словарь с результатами
    asr_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @property
    def audio_path_obj(self) -> Path:
        return Path(self.audio_path)

    @property
    def has_error(self) -> bool:
        """True, если любая модель записала ошибку для этого сэмпла."""
        return any(bool(res.get("error")) for res in self.asr_results.values())
    
    def add_result(self, model_name: str, hypothesis: str, duration: float = 0.0, error: Optional[str] = None, **kwargs):
        """
        Добавляет результат модели в историю.
        Не сохраняйте сюда тяжелые данные (аудио, эмбеддинги)!
        """
        # Валидация тяжелых данных
        for k, v in kwargs.items():
            try:
                if sys.getsizeof(v) > 50 * 1024: # 50KB limit warning
                    print(f"⚠️ Warning: Large object in result '{k}' for model '{model_name}' ({sys.getsizeof(v)} bytes). Avoid storing embeddings or audio data in JSONL.")
            except:
                pass

        result_data = {
            "hypothesis": hypothesis,
            "processing_time": duration, # Время инференса
            "error": error,
            **kwargs
        }
        self.asr_results[model_name] = result_data

    def get_best_result(self, metric: str = 'wer', mode: str = 'min') -> Tuple[Optional[str], float]:
        """
        Находит модель с лучшим значением указанной метрики для этого сэмпла.
        
        Args:
            metric: Имя метрики (wer, cer, accuracy...)
            mode: 'min' для ошибок (WER/CER), 'max' для точности (Accuracy)
            
        Returns:
            (model_name, score). Если метрик нет, возвращает (None, inf/-inf).
        """
        best_model = None
        best_score = float('inf') if mode == 'min' else float('-inf')
        
        for model, res in self.asr_results.items():
            if 'metrics' in res and res['metrics'] and metric in res['metrics']:
                score = res['metrics'][metric]
                
                if mode == 'min':
                    if score < best_score:
                        best_score = score
                        best_model = model
                else: # mode == 'max'
                    if score > best_score:
                        best_score = score
                        best_model = model
                        
        return best_model, best_score

    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь."""
        d = asdict(self)
        meta = d.pop("meta", {}) or {}

        # Исторически meta "расплющивалась" в корень для удобного анализа в pandas.
        # Делаем это безопасно: при коллизиях не затираем системные поля.
        for k, v in meta.items():
            if k in d:
                print(
                    f"⚠️ Warning: meta key '{k}' conflicts with top-level field in AudioSample. "
                    f"Storing it as 'meta__{k}'."
                )
                d[f"meta__{k}"] = v
            else:
                d[k] = v

        return d
