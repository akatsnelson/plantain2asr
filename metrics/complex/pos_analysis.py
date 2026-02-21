import pymorphy2
import inspect
from collections import namedtuple

# Monkey-patch для совместимости pymorphy2 с Python 3.11+
if not hasattr(inspect, 'getargspec'):
    ArgSpec = namedtuple("ArgSpec", ["args", "varargs", "keywords", "defaults"])
    def getargspec(func):
        spec = inspect.getfullargspec(func)
        return ArgSpec(spec.args, spec.varargs, spec.varkw, spec.defaults)
    inspect.getargspec = getargspec

from collections import defaultdict
from typing import Dict, List
from ..base import BaseMetric
from ..utils.alignment import align_words

class PosErrorAnalysis(BaseMetric):
    """
    Анализирует ошибки (WER) в разрезе частей речи (POS).
    Использует pymorphy2 для определения частей речи.
    """
    
    def __init__(self, do_clean: bool = True):
        super().__init__(do_clean)
        self.morph = pymorphy2.MorphAnalyzer()
        
        # Маппинг тегов pymorphy2 в человекочитаемые
        self.pos_names = {
            'NOUN': 'Существительное',
            'ADJF': 'Прилагательное',
            'ADJS': 'Прилагательное (кр)',
            'COMP': 'Компаратив',
            'VERB': 'Глагол',
            'INFN': 'Глагол (инф)',
            'PRTF': 'Причастие',
            'PRTS': 'Причастие (кр)',
            'GRND': 'Деепричастие',
            'NUMR': 'Числительное',
            'ADVB': 'Наречие',
            'NPRO': 'Местоимение',
            'PRED': 'Предикатив',
            'PREP': 'Предлог',
            'CONJ': 'Союз',
            'PRCL': 'Частица',
            'INTJ': 'Междометие'
        }

    @property
    def name(self) -> str:
        return "PosErrorRate"

    def _get_pos(self, word: str) -> str:
        """Определяет часть речи слова"""
        p = self.morph.parse(word)[0]
        tag = p.tag.POS
        return self.pos_names.get(tag, str(tag)) if tag else "Other"

    def calculate(self, reference: str, hypothesis: str) -> Dict[str, dict]:
        """
        Возвращает статистику по частям речи.
        """
        if self.do_clean:
            reference = self.normalize(reference)
            hypothesis = self.normalize(hypothesis)
            
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        
        alignment = align_words(ref_words, hyp_words)
        
        stats = defaultdict(lambda: {"total": 0, "errors": 0, "correct": 0, "sub": 0, "del": 0})
        
        for op, ref_w, hyp_w in alignment:
            if op == "ins":
                # Вставки не имеют POS в референсе, пропускаем или считаем в Other
                continue
                
            pos = self._get_pos(ref_w)
            stats[pos]["total"] += 1
            
            if op == "correct":
                stats[pos]["correct"] += 1
            else:
                stats[pos]["errors"] += 1
                if op == "sub":
                    stats[pos]["sub"] += 1
                elif op == "del":
                    stats[pos]["del"] += 1
        
        # Считаем Error Rate для каждого POS
        result = {}
        for pos, data in stats.items():
            error_rate = (data["errors"] / data["total"] * 100.0) if data["total"] > 0 else 0.0
            result[pos] = {
                "ErrorRate": error_rate,
                **data
            }
            
        return dict(result)

    def calculate_batch(self, references: List[str], hypotheses: List[str]) -> Dict[str, dict]:
        """Агрегирует статистику по батчу"""
        agg_stats = defaultdict(lambda: {"total": 0, "errors": 0, "correct": 0, "sub": 0, "del": 0})
        
        for ref, hyp in zip(references, hypotheses):
            # Получаем стат для одного предложения
            sample_stats = self.calculate(ref, hyp)
            
            # Агрегируем
            for pos, metrics in sample_stats.items():
                agg_stats[pos]["total"] += metrics["total"]
                agg_stats[pos]["errors"] += metrics["errors"]
                agg_stats[pos]["correct"] += metrics["correct"]
                agg_stats[pos]["sub"] += metrics["sub"]
                agg_stats[pos]["del"] += metrics["del"]
        
        # Финальный расчет процентов
        result = {}
        for pos, data in agg_stats.items():
            error_rate = (data["errors"] / data["total"] * 100.0) if data["total"] > 0 else 0.0
            result[pos] = {
                "ErrorRate": error_rate,
                "Count": data["total"],
                "Correct": data["correct"],
                "Errors": data["errors"]
            }
            
        return dict(result)
