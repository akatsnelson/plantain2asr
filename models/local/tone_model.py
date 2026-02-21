import sys
import os
from pathlib import Path
from typing import Union
from ..base import BaseASRModel

# T-one - специфичная библиотека
# Оборачиваем импорт в try-except Exception, так как могут быть ошибки 
# не только ImportError (например, конфликт версий numpy/onnxruntime)
# ВАЖНО: Импорты отложены в __init__, чтобы ошибки не блокировали весь модуль

tone = None
ort = None
StreamingCTCPipeline = None
onnx_wrapper = None

class ToneModel(BaseASRModel):
    def __init__(self, model_name: str = None, device: str = "cuda"):
        """
        :param model_name: Путь к локальной модели. Если None, загружает из Hugging Face.
        """
        # Отложенный импорт
        global tone, ort, StreamingCTCPipeline, onnx_wrapper
        
        if tone is None:
            try:
                import tone as tone_lib
                import onnxruntime as ort_lib
                from tone import StreamingCTCPipeline as SCP
                import tone.onnx_wrapper as ow
                
                tone = tone_lib
                ort = ort_lib
                StreamingCTCPipeline = SCP
                onnx_wrapper = ow
            except Exception as e:
                raise ImportError(f"tone library not available: {e}")
            
        self._name = "T-One"
        self.device = device # Just stored, used implicitly via ONNX providers
        
        # === ONNX Providers Patch ===
        # Переопределяем метод создания сессии, чтобы инжектировать провайдеры (CUDA/CoreML)
        def from_local_accelerated(cls, model_path):
            providers = ort.get_available_providers()
            active_providers = []
            
            if 'CUDAExecutionProvider' in providers and (device == "cuda" or device == "auto"):
                print("   ✅ T-One using CUDA")
                active_providers.append('CUDAExecutionProvider')
                
            if 'CoreMLExecutionProvider' in providers and (device == "mps" or device == "auto"):
                print("   ✅ T-One using CoreML (MPS)")
                active_providers.append('CoreMLExecutionProvider')
                
            active_providers.append('CPUExecutionProvider')
            
            sess = ort.InferenceSession(model_path, providers=active_providers)
            return cls(sess)
            
        if hasattr(onnx_wrapper, 'StreamingCTCModel'):
            onnx_wrapper.StreamingCTCModel.from_local = classmethod(from_local_accelerated)
        # ==================
        
        if model_name and os.path.exists(model_name):
            print(f"📥 Loading T-One from local path: {model_name}...")
            self.pipeline = StreamingCTCPipeline.from_local(model_name)
        else:
            print(f"📥 Loading T-One from HuggingFace (default)...")
            self.pipeline = StreamingCTCPipeline.from_hugging_face()

    @property
    def name(self) -> str:
        return self._name

    def transcribe(self, audio_path: Union[str, Path]) -> str:
        result = self.pipeline(str(audio_path))
        
        if hasattr(result, 'text'):
            return result.text
        elif isinstance(result, list):
            return " ".join([p.text for p in result if hasattr(p, 'text')])
        
        return str(result)
