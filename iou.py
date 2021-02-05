import torch
from torch.autograd import Function


class Iou(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) - self.inter

        t = self.inter/self.union

        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def iou_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()

    else:
        s = torch.FloatTensor(1).zero_()


    for i, c in enumerate(zip(input, target)):
        s = s + Iou().forward(c[0], c[1])

    return s / (i + 1)


def pres_recall(input, target):
    tp = (target * input).sum().to(torch.float32)
    tn = ((1 - target) * (1 - input)).sum().to(torch.float32)
    fp = ((1 - target) * input).sum().to(torch.float32)
    fn = (target * (1 - input)).sum().to(torch.float32)

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2* (precision*recall) / (precision + recall + epsilon)
    return precision, recall, f1