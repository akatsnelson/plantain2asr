import torch
import os
from typing import Union, List, Optional, Tuple, Any
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from ..base import BaseASRModel
from transformers import AutoModel, AutoProcessor

class GigaAMv3(BaseASRModel):
    """
    Обертка для GigaAM v3 (HuggingFace AutoModel) с оптимизированным батчингом.
    
    Особенности:
    - Адаптивная стратегия: GPU/MPS батчинг или CPU многопоточность
    - Параллельная загрузка файлов
    - Динамическое разбиение на подбатчи для защиты от OOM
    - Graceful error handling
    """
    def __init__(self, model_name: str = "e2e_rnnt", device: str = "cuda"):
        if AutoModel is None:
            raise ImportError("transformers not installed")
            
        # Smart device selection
        if device == "cuda" and not torch.cuda.is_available():
            if torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        
        self._name = f"GigaAM-v3-{model_name}"
        
        print(f"📥 Loading GigaAM-v3 ({model_name}) from HF on {self.device}...")
        
        self.model = AutoModel.from_pretrained(
            "ai-sage/GigaAM-v3",
            revision=model_name,
            trust_remote_code=True,
            low_cpu_mem_usage=False,  # torchaudio.MelScale несовместима с meta tensors
            device_map=None,          # отключает авто-определение device (тоже триггерит meta)
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Определяем внутреннюю модель для доступа к низкоуровневым методам (prepare_wav, forward)
        # AutoModel возвращает обертку, внутри которой лежит реальная модель в атрибуте .model
        if hasattr(self.model, "model") and hasattr(self.model.model, "prepare_wav"):
            self._inner_model = self.model.model
        else:
            self._inner_model = self.model

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_e2e(self) -> bool:
        return "e2e" in self._name.lower()

    def transcribe(self, audio_path: Union[str, Path]) -> str:
        # transcribe доступен на верхней обертке
        return self.model.transcribe(str(audio_path)).strip()

    def transcribe_batch(
        self, 
        audio_paths: List[Union[str, Path]], 
        num_workers: Optional[int] = None,
        max_gpu_batch_size: int = 32
    ) -> List[str]:
        """
        Батчевая транскрипция аудиофайлов с адаптивной стратегией.
        """
        if not audio_paths:
            return []
        
        # Используем "GPU-стратегию" (батчинг) для CUDA и MPS
        use_batching = self.device != "cpu"
        
        if use_batching:
            return self._transcribe_batch_gpu(audio_paths, max_gpu_batch_size)
        else:
            if num_workers is None:
                num_workers = min(len(audio_paths), os.cpu_count() or 4)
            return self._transcribe_batch_cpu(audio_paths, num_workers)
    
    def _transcribe_batch_gpu(
        self, 
        audio_paths: List[Union[str, Path]], 
        max_batch_size: int
    ) -> List[str]:
        """
        Оптимизированный батчинг (CUDA/MPS).
        """
        if len(audio_paths) > max_batch_size:
            all_transcriptions = []
            for i in range(0, len(audio_paths), max_batch_size):
                sub_batch = audio_paths[i:i + max_batch_size]
                all_transcriptions.extend(self._transcribe_single_gpu_batch(sub_batch))
            return all_transcriptions
        else:
            return self._transcribe_single_gpu_batch(audio_paths)
    
    def _transcribe_single_gpu_batch(self, audio_paths: List[Union[str, Path]]) -> List[str]:
        """
        Обработка одного батча.
        """
        def load_audio_file(path):
            try:
                # Используем _inner_model для доступа к prepare_wav
                return self._inner_model.prepare_wav(str(path))
            except Exception as e:
                print(f"⚠️ Ошибка загрузки {path}: {e}")
                return None
        
        with ThreadPoolExecutor(max_workers=min(8, len(audio_paths))) as executor:
            prepared = list(executor.map(load_audio_file, audio_paths))
        
        valid_indices = [i for i, p in enumerate(prepared) if p is not None]
        if not valid_indices:
            return [""] * len(audio_paths)
        
        wavs = []
        lengths = []
        for i in valid_indices:
            wav, length = prepared[i]
            wavs.append(wav.squeeze(0))
            lengths.append(length.item())
        
        max_len = max(lengths)
        batch_wav = torch.zeros(len(wavs), max_len, device=self.device, dtype=wavs[0].dtype)
        batch_lengths = torch.tensor(lengths, device=self.device)
        
        for i, wav in enumerate(wavs):
            batch_wav[i, :len(wav)] = wav
        
        with torch.no_grad():
            # Используем _inner_model для forward и decoding
            encoded, encoded_len = self._inner_model.forward(batch_wav, batch_lengths)
            transcriptions = self._inner_model.decoding.decode(self._inner_model.head, encoded, encoded_len)
        
        results = []
        valid_idx = 0
        for i in range(len(audio_paths)):
            if i in valid_indices:
                results.append(transcriptions[valid_idx].strip())
                valid_idx += 1
            else:
                results.append("")
        
        return results
    
    def _transcribe_batch_cpu(self, audio_paths: List[Union[str, Path]], num_workers: int) -> List[str]:
        """
        Оптимизированная параллельная обработка на CPU.
        """
        results = [None] * len(audio_paths)
        
        def transcribe_single(idx: int, path: Union[str, Path]) -> tuple:
            try:
                transcription = self.model.transcribe(str(path)).strip()
                return (idx, transcription, None)
            except Exception as e:
                return (idx, "", str(e))
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(transcribe_single, i, path)
                for i, path in enumerate(audio_paths)
            ]
            
            for future in as_completed(futures):
                idx, transcription, error = future.result()
                results[idx] = transcription
                if error:
                    print(f"⚠️ Ошибка при транскрипции {audio_paths[idx]}: {error}")
        
        return results

    def get_training_components(self) -> Tuple[Any, Any, Any]:
        """
        Возвращает компоненты для обучения модели GigaAM.
        """
        print("🔧 Preparing training components for GigaAM-v3...")
        
        processor = None
        
        # 1. Загружаем процессор/токенизатор
        # Попытка 1: Стандартный AutoProcessor (обычно падает для кастомных моделей)
        try:
            processor = AutoProcessor.from_pretrained("ai-sage/GigaAM-v3", trust_remote_code=True)
        except Exception:
            pass
            
        # Попытка 2: Ищем в атрибутах модели
        if processor is None:
            # Рекурсивный поиск токенизатора
            def find_attr(obj, name, depth=0, max_depth=2):
                if depth > max_depth: return None
                if hasattr(obj, name): return getattr(obj, name)
                if hasattr(obj, "__dict__"):
                    for k, v in obj.__dict__.items():
                        if not k.startswith("_"): # Skip private
                            res = find_attr(v, name, depth+1, max_depth)
                            if res: return res
                return None
            
            tokenizer = find_attr(self.model, "tokenizer") or find_attr(self._inner_model, "tokenizer")
            
            if tokenizer:
                # Если нашли токенизатор, нам все равно нужен Processor (FeatureExtractor + Tokenizer)
                # Создаем гибридный Wav2Vec2Processor, подменяя токенизатор
                from transformers import Wav2Vec2Processor, Wav2Vec2FeatureExtractor
                try:
                    # Берем стандартный экстрактор
                    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
                    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
                except Exception as e:
                    print(f"⚠️ Could not wrap custom tokenizer into Wav2Vec2Processor: {e}")
                    processor = tokenizer # Возвращаем хотя бы токенизатор (но feature extraction сломается)

        # Попытка 3: Фоллбек на стандартный русский токенизатор (ОПАСНО: словари могут не совпасть!)
        if processor is None:
            print("⚠️ WARNING: Could not find GigaAM tokenizer. Using fallback 'jonatasgrosman/wav2vec2-large-xlsr-53-russian'.")
            print("⚠️ NOTE: Fine-tuning might fail if vocabularies mismatch!")
            from transformers import Wav2Vec2Processor
            processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-russian")

        # 2. Подготовка модели к обучению
        self.model.train()
        # По умолчанию замораживаем Feature Extractor (аудио-энкодер), учим только трансформер
        if hasattr(self.model, "freeze_feature_encoder"):
            self.model.freeze_feature_encoder()
        elif hasattr(self._inner_model, "freeze_feature_encoder"):
            self._inner_model.freeze_feature_encoder()
            
        print("   ✅ Model is in TRAIN mode (feature extractor frozen)")

        # 3. Data Collator (возвращаем None, Trainer должен использовать дефолтный или мы добавим позже)
        return self.model, processor, None
