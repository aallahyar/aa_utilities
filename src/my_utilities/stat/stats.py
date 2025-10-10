
class IterativeStats:
    def __init__(self):
        self.count = 0
        self.mean_ = 0.0
        self.M2_ = 0.0

    def include(self, value):
        # Welford's online algorithm
        self.count += 1
        delta = value - self.mean_
        self.mean_ += delta / self.count
        delta2 = value - self.mean_
        self.M2_ += delta * delta2

    def mean(self):
        return self.mean_

    def var(self):
        return self.M2_ / self.count if self.count > 0 else float('nan')

    def std(self):
        return (self.var()) ** 0.5

if __name__ == '__main__':
    import numpy as np
    rng = np.random.default_rng(seed=42)
    data = rng.integers(-100, 100, size=1000)
    # print(data)

    stats = IterativeStats()
    for x in data:
        stats.include(x)
    print(data.mean(), stats.mean(), data.mean() - stats.mean())
    print(data.var(), stats.var(), data.var() - stats.var())
    print(data.std(), stats.std(), data.std() - stats.std())