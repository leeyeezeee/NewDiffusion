class Accuracy:
    """
    Simple accuracy tracker.
    """
    def __init__(self):
        self._num_correct = 0
        self._num_total = 0

    def update(self, predicted: str, target: str) -> bool:
        is_correct = predicted == target
        self._num_correct += int(is_correct)
        self._num_total += 1
        return is_correct

    def get(self) -> float:
        return self._num_correct / self._num_total if self._num_total > 0 else 0.0

    def print(self):
        acc = self.get() * 100
        print(f"Accuracy: {acc:.1f}% ({self._num_correct}/{self._num_total})")
