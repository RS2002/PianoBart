from custom.parallel import DataParallelCriterion

import torch
import numpy as np
import torch.nn.functional as F

from typing import Dict

import shapesimilarity


class _Metric(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        raise NotImplementedError()

class _FAD(_Metric):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        NOTE=128
        gap=10
        FAD=0
        for i in range(input.shape[0]):
            current_FAD=0
            y1=input[i][input[i]<NOTE].tolist()
            y2=target[i][target[i]<NOTE].tolist()
            l1=len(y1)
            l2=len(y2)
            if l1>l2:
                y1=y1[:l2]
            elif l2>l1:
                y2=y2[:l1]
            for j in range(len(y1)//gap):
                c1 = y1[i * gap:(i + 1) * gap - 1]
                c2 = y2[i * gap:(i + 1) * gap - 1]
                x = range(gap)
                current_FAD += shapesimilarity.shape_similarity(list(zip(x, c1)), list(zip(x, c2)))
            FAD+=current_FAD/(len(y1)//gap)
        FAD/=input.shape[0]
        return FAD


class FAD(_FAD):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """
        :param input: [B, T, V]
        :param target: [B, T]
        :return:
        """
        input = input.softmax(-1)
        categorical_input = input.argmax(-1)
        return super().forward(categorical_input, target)


class Accuracy(_Metric):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """
        :param input: [B, L]
        :param target: [B, L]
        :return:
        """
        bool_acc = input.long() == target.long()
        return bool_acc.sum().to(torch.float) / bool_acc.numel()


class MockAccuracy(Accuracy):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        return super().forward(input, target)


class CategoricalAccuracy(Accuracy):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """
        :param input: [B, T, V]
        :param target: [B, T]
        :return:
        """
        input = input.softmax(-1)
        categorical_input = input.argmax(-1)
        return super().forward(categorical_input, target)


class LogitsBucketting(_Metric):
    def __init__(self, vocab_size):
        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        return input.argmax(-1).flatten().to(torch.int32)


class MetricsSet(object):
    def __init__(self, metric_dict: Dict):
        super().__init__()
        self.metrics = metric_dict

    def __call__(self, input: torch.Tensor, target: torch.Tensor):
        return self.forward(input=input, target=target)

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        # return [metric(input, target) for metric in self.metrics]
        return {
            k: metric(input.to(target.device), target)
            for k, metric in self.metrics.items()}


class ParallelMetricSet(MetricsSet):
    def __init__(self, metric_dict: Dict):
        super(ParallelMetricSet, self).__init__(metric_dict)
        self.metrics = {k: DataParallelCriterion(v) for k, v in metric_dict.items()}

    def forward(self, input, target):
        # return [metric(input, target) for metric in self.metrics]
        return {
            k: metric(input, target)
            for k, metric in self.metrics.items()}



if __name__ == '__main__':
    met = MockAccuracy()
    test_tensor1 = torch.ones((3,2)).contiguous().cuda().to(non_blocking=True, dtype=torch.int)
    test_tensor2 = torch.ones((3,2)).contiguous().cuda().to(non_blocking=True, dtype=torch.int)
    test_tensor3 = torch.zeros((3,2))
    print(met(test_tensor1, test_tensor2))
