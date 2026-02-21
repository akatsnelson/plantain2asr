from .mer import MER

class Accuracy(MER):
    """
    Accuracy (Word Match Rate) = 1 - MER.
    Показывает долю правильно распознанных слов (Matches) от общего числа операций.
    """
    @property
    def name(self) -> str:
        return "Accuracy"

    def calculate(self, reference: str, hypothesis: str) -> float:
        mer = super().calculate(reference, hypothesis)
        return 100.0 - mer

    def calculate_batch(self, references: list, hypotheses: list) -> float:
        mer = super().calculate_batch(references, hypotheses)
        return 100.0 - mer
