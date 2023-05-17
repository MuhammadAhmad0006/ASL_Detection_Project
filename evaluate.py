import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.autograd import Variable
import onnx
import onnxruntime as ort
from dataset import get_train_test_loaders
from train import cnnModel

#standared evaluatore that compared against nonone-hot encoded variables
def evaluate(outputs: Variable, labels: Variable) -> float:
    Y = labels.numpy()
    Yhat = np.argmax(outputs, axis=1) # prediction variable for classes 
    return float(np.sum(Yhat == Y)) # comapres actual answer to the predicted answer 

#batch evaluation for batches of data that large 
def batch_evaluate(
        net: cnnModel,
        dataloader: torch.utils.data.DataLoader) -> float:
    score = n = 0.0
    for batch in dataloader:
        n += len(batch['image'])
        outputs = net(batch['image'])
        if isinstance(outputs, torch.Tensor):
            outputs = outputs.detach().numpy()
        score += evaluate(outputs, batch['label'][:, 0])
    return score / n


def validation():
    trainloader, testloader = get_train_test_loaders()
    net = cnnModel().float().eval()

    pretrained_model = torch.load("saveModel.pth")
    net.load_state_dict(pretrained_model)

    print('PyTorch')
    train_acc = batch_evaluate(net, trainloader) * 100.
    print('Training accuracy is: %.1f' % train_acc)
    test_acc = batch_evaluate(net, testloader) * 100.
    print('Validation accuracy is: %.1f' % test_acc)

    trainloader, testloader = get_train_test_loaders(1)

    # export to onnx 
    fname = "signlanguage.onnx"
    dummy = torch.randn(1, 1, 28, 28)
    torch.onnx.export(net, dummy, fname, input_names=['input'])

    model = onnx.load(fname)
    onnx.checker.check_model(model) 

    ort_session = ort.InferenceSession(fname)
    net = lambda inp: ort_session.run(None, {'input': inp.data.numpy()})[0]

    print('ONNX')
    train_acc = batch_evaluate(net, trainloader) * 100.
    print('Training accuracy is: %.1f' % train_acc)
    test_acc = batch_evaluate(net, testloader) * 100.
    print('Validation accuracy is: %.1f' % test_acc)


if __name__ == '__main__':
    validation()