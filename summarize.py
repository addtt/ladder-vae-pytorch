from collections import deque

import numpy as np


class History:
    """
    Wrapper of SummarizerCollection to register new history values with timestamp.
    """
    def __init__(self):
        self.summarizers = SummarizerCollection(mode='store_all')

    def add(self, summaries, timestamp):
        """
        Adds new summaries with a timestamp.

        :param summaries: dictionary of summaries. E.g. key is the name of the
                    quantity, value is its average over the last epoch.
        :param timestamp:
        """
        timestamped_dict = {k: (timestamp, v) for k, v in summaries.items()}
        self.summarizers.add(timestamped_dict)

    def get_dict(self):
        return self.summarizers

    def __str__(self):
        return str(self.summarizers)


class SummarizerCollection:
    def __init__(self, mode, ma_length=None):
        legal_modes = [
            'total_average',
            'sum',
            'moving_average',
            'store_all',
        ]
        if mode not in legal_modes:
            raise RuntimeError("unrecognized mode '{}'".format(mode))
        assert (ma_length is not None) == (mode == 'moving_average')
        self.ma_length = ma_length
        self.mode = mode
        self.summarizers = {}   # dict of summarizers

    def _new_summarizer(self):
        if self.mode == 'store_all':
            return HistorySummarizer()
        if self.mode == 'total_average':
            return TotalAverageSummarizer()
        if self.mode == 'sum':
            return SumSummarizer()
        if self.mode == 'moving_average':
            return MASummarizer(self.ma_length)

    def add(self, data):
        """
        Update the summarizers with new values.
        :param data: a dictionary
        """

        for name, value in data.items():
            if name not in self.summarizers:
                self.summarizers[name] = self._new_summarizer()
            self.summarizers[name].add(value)

    def get(self, name):
        """
        Returns the summary of the quantity with the given name.
        :param name: the quantity's name
        :return: the summary of the quantity
        """
        return self.summarizers[name].get()

    def get_all(self, reset=False):
        """
        Returns a dict of all summaries, where keys are the names and values
        are the corresponding summaries. Optionally resets the summarizer.
        Dict is a (shallow) copy of the internal dict.
        """
        out = {n: self[n] for n in self}
        if reset:
            self.reset()
        return out

    def __getitem__(self, name):
        return self.get(name)

    def __contains__(self, name):
        return name in self.summarizers

    def __iter__(self):
        for name in self.summarizers:
            yield name

    def __str__(self):
        s = type(self).__name__ + "(\n"
        s += "  mode: {}\n".format(self.mode)
        if self.ma_length is not None:
            s += "  ma_length: {}\n".format(self.ma_length)
        s += "  summarizers:\n"
        for name in self:
            s += "    {}: {}\n".format(name, self.summarizers[name])
        return s + ")"

    def reset(self):
        """
        Re-initializes the object. This should be called after getting the summary of all
        quantities, before starting to collect the new values for the next summary.
        """
        self.__init__(self.mode, self.ma_length)


class Summarizer:
    """
    Base class for Summarizers. A Summarizer provides a simple way to get
    a summary of a sequence of values. The specific summary depends on the
    implementation of the Summarizer.

    A Summarizer is updated by adding new values to it. It keeps a state that
    represents the summary of values added so far, so the method get() simply
    returns the internal state.

    For now, a summary can be the full history, the overall average, a moving
    average with a limited window, or the sum of all values.
    """
    def __init__(self):
        self.state = None

    def add(self, value):
        pass

    def get(self):
        return self.state


class HistorySummarizer(Summarizer):
    def __init__(self):
        super().__init__()
        self.state = []

    def add(self, value):
        self.state.append(value)

    def __str__(self):
        if len(self.state) < 4:
            return ", ".join(str(v) for v in self.state)
        return "{}, ..., {}".format(self.state[0], self.state[-1])


class TotalAverageSummarizer(Summarizer):
    def __init__(self):
        super().__init__()
        self.n_items = 0
        self.state = 0.0

    def add(self, value):
        self.n_items += 1
        self.state += (value - self.state) / self.n_items

    def __str__(self):
        return "{} ({} items)".format(self.state, self.n_items)


class SumSummarizer(Summarizer):
    def __init__(self):
        super().__init__()
        self.n_items = 0
        self.state = 0.0

    def add(self, value):
        self.n_items += 1
        self.state += value

    def __str__(self):
        return "{} ({} items)".format(self.state, self.n_items)


class MASummarizer(Summarizer):
    def __init__(self, ma_length):
        super().__init__()
        self.ma_length = ma_length
        self.list = deque(maxlen=ma_length)
        self.state = 0.0

    def add(self, value):
        n = len(self.list)
        if n == self.ma_length:
            self.state += (value - self.list[0]) / n
        else:  # n < ma_length
            # always keeps average of numbers in memory, even
            # when there's fewer than ma_length
            self.state += (value - self.state) / (n + 1)
        self.list.append(value)

    def __str__(self):
        return "{}".format(self.state)


if __name__ == '__main__':
    summ = MASummarizer(4)
    for i in range(10):
        summ.add(3)
        assert summ.get() == 3
        assert len(summ.list) == min(i + 1, summ.ma_length)

    summ = TotalAverageSummarizer()
    for i in range(10):
        summ.add(3)
        assert summ.get() == 3
        assert summ.n_items == i + 1

    summ = MASummarizer(20)
    for i in range(50):
        summ.add(np.random.rand())
        assert abs(summ.get() - np.mean(summ.list)) < 1e-8, \
            "{} != {}".format(summ.get(), np.mean(summ.list))
        assert len(summ.list) == min(i + 1, summ.ma_length)
