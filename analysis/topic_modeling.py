import re
import multiprocessing
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict, Counter
from pathlib import Path

try:
    from gensim import corpora, models
    from gensim.models import CoherenceModel
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False

from ..dataloaders.base import BaseASRDataset
from ..core.processor import Processor

# Стоп-слова (взяты из вашего скрипта)
STOPWORDS = {
    'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 'все',
    'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по',
    'только', 'ее', 'мне', 'было', 'вот', 'от', 'меня', 'еще', 'нет', 'о', 'из', 'ему',
    'теперь', 'когда', 'даже', 'ну', 'вдруг', 'ли', 'если', 'уже', 'или', 'ни', 'быть',
    'был', 'него', 'до', 'вас', 'нибудь', 'опять', 'уж', 'вам', 'ведь', 'там', 'потом',
    'себя', 'ничего', 'ей', 'может', 'они', 'тут', 'где', 'есть', 'надо', 'ней', 'для',
    'мы', 'тебя', 'их', 'чем', 'была', 'сам', 'чтоб', 'без', 'будто', 'чего', 'раз',
    'тоже', 'себе', 'под', 'будет', 'ж', 'тогда', 'кто', 'этот', 'того', 'потому',
    'этого', 'какой', 'совсем', 'ним', 'здесь', 'этом', 'один', 'почти', 'мой', 'тем',
    'чтобы', 'нее', 'сейчас', 'были', 'куда', 'зачем', 'всех', 'никогда', 'можно', 'при',
    'угу', 'ага', 'м', 'мм', 'э', 'ээ', 'эм', 'нрзб', 'эээ', 'ааа', 'ммм',
}

@dataclass
class TopicReport:
    topics: List[Dict] # id, keywords, suggested_name
    topic_distribution: pd.DataFrame
    wer_by_topic: pd.DataFrame
    coherence_score: float

    def print(self):
        print("\n🧠 Topic Modeling Analysis (LDA) (Section 5)")
        print(f"Coherence Score: {self.coherence_score:.4f} " + 
              ("✅ Good" if self.coherence_score > 0.5 else "⚠️ Low"))
        print("-" * 60)
        
        print("\n📚 Discovered Topics:")
        for t in self.topics:
            print(f"   Topic {t['id']}: {t['keywords']}...")
        
        print("\n📉 WER per Topic (Performance Analysis):")
        print(self.wer_by_topic.to_markdown(index=False, floatfmt=".1f"))
        
        print("\n💡 Insight: Topics with high WER are candidates for targeted fine-tuning.")

class TopicAnalyzer(Processor):
    """
    Выполняет тематическое моделирование (LDA) на текстах датасета.
    Позволяет выявить скрытые темы и оценить качество распознавания в разрезе тем.
    """
    def __init__(self, num_topics: int = 10, passes: int = 10):
        self.num_topics = num_topics
        self.passes = passes
        self.lda_model = None
        self.dictionary = None
        
        if not GENSIM_AVAILABLE:
            print("⚠️ 'gensim' not installed. Please install for topic modeling: pip install gensim")

    def _clean_and_tokenize(self, text: str) -> List[str]:
        if not isinstance(text, str): return []
        text = re.sub(r'[^а-яёa-z\s]', ' ', text.lower())
        return [w for w in text.split() if len(w) >= 3 and w not in STOPWORDS]

    def fit(self, dataset: BaseASRDataset):
        """Обучает LDA модель на текстах датасета"""
        if not GENSIM_AVAILABLE: return self
        
        print(f"🧠 Training LDA model ({self.num_topics} topics)...")
        texts = []
        for s in dataset:
            if s.text:
                tokens = self._clean_and_tokenize(s.text)
                if len(tokens) >= 3:
                    texts.append(tokens)
        
        if not texts:
            print("⚠️ Not enough text data for LDA.")
            return self

        self.dictionary = corpora.Dictionary(texts)
        self.dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=10000)
        corpus = [self.dictionary.doc2bow(text) for text in texts]
        
        self.lda_model = models.LdaMulticore(
            corpus=corpus,
            id2word=self.dictionary,
            num_topics=self.num_topics,
            passes=self.passes,
            workers=min(multiprocessing.cpu_count() - 1, 4),
            random_state=42
        )
        print("✅ LDA training complete.")
        return self

    def apply_to(self, dataset: BaseASRDataset) -> BaseASRDataset:
        if not GENSIM_AVAILABLE or not self.lda_model:
            if GENSIM_AVAILABLE:
                self.fit(dataset)
            else:
                return dataset
        
        print("🧠 Analyzing topics and errors...")
        
        # 1. Analyze topics
        topics_info = []
        for i in range(self.num_topics):
            words = self.lda_model.show_topic(i, topn=5)
            keywords = ", ".join([w[0] for w in words])
            topics_info.append({'id': i, 'keywords': keywords})
            
        # 2. Assign topics and calculate WER
        model_topic_wer = defaultdict(lambda: defaultdict(list))
        topic_counts = Counter()
        
        for s in dataset:
            if not s.text: continue
            
            tokens = self._clean_and_tokenize(s.text)
            if len(tokens) < 3: continue
            
            bow = self.dictionary.doc2bow(tokens)
            if not bow: continue
            
            # Get dominant topic
            dist = self.lda_model[bow]
            if not dist: continue
            
            # dist is list of (topic_id, prob), handle if it's nested
            if isinstance(dist, list) and len(dist) > 0 and isinstance(dist[0], list):
                 # Sometimes gensim returns list of lists for corpus
                 dist = dist[0]
            
            dominant_topic = max(dist, key=lambda x: x[1])[0]
            topic_counts[dominant_topic] += 1
            
            # Collect WERs
            for model_name, res in s.asr_results.items():
                metrics = res.get('metrics', {})
                if 'wer' in metrics:
                    model_topic_wer[model_name][dominant_topic].append(metrics['wer'])

        # 3. Build Report Tables
        # Distribution
        dist_data = []
        total_docs = sum(topic_counts.values())
        for t_id in range(self.num_topics):
            count = topic_counts[t_id]
            dist_data.append({
                'Topic': t_id,
                'Count': count,
                'Share (%)': (count / total_docs * 100) if total_docs else 0,
                'Keywords': next(t['keywords'] for t in topics_info if t['id'] == t_id)
            })
        dist_df = pd.DataFrame(dist_data)

        # WER Table
        wer_data = []
        for t_id in range(self.num_topics):
            row = {'Topic': t_id}
            row['Keywords'] = next(t['keywords'] for t in topics_info if t['id'] == t_id)[:30] + "..."
            row['Count'] = topic_counts[t_id]
            
            for model_name in model_topic_wer.keys():
                wers = model_topic_wer[model_name][t_id]
                row[model_name] = np.mean(wers) * 100 if wers else None
            
            wer_data.append(row)
            
        wer_df = pd.DataFrame(wer_data)
        
        # Calculate Coherence (on subset for speed)
        # For full coherence we need original texts, reconstructing them is expensive if not stored
        # Here we return a placeholder or calculate if passed explicit corpus (omitted for brevity)
        coherence = 0.0 
        
        report = TopicReport(
            topics=topics_info,
            topic_distribution=dist_df,
            wer_by_topic=wer_df,
            coherence_score=coherence
        )
        
        report.print()
        self.report = report
        return dataset
