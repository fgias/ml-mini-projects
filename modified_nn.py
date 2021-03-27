import torch
from torchvision import datasets
import matplotlib.pyplot as plt
mnist = datasets.MNIST('./data', download=True)

zeroes = mnist.data[(mnist.targets == 0)]/255.0
ones = mnist.data[(mnist.targets == 1)]/255.0

len(zeroes), len(ones)

def show_image(img):
  plt.imshow(img)
  plt.xticks([])
  plt.yticks([])
  plt.show()

combined_data = torch.cat([zeroes, ones])
combined_data.shape

flat_imgs = combined_data.view(-1, 28*28)
flat_imgs.shape

target = torch.tensor([0]*len(zeroes)+[1]*len(ones))
target.shape

def sigmoid(x): return 1/(1+torch.exp(-x))

def simple_nn(data, weights, bias): return sigmoid((data@weights) + bias)

def error(pred, target): return ((pred-target)**2).mean()

w = torch.randn((flat_imgs.shape[1], 1), requires_grad=True)
b = torch.randn((1, 1), requires_grad=True)

for i in range(5000):
  pred = simple_nn(flat_imgs, w, b)
  loss = error(pred, target.unsqueeze(1))
  loss.backward()

  w.data -= 0.001*w.grad.data
  b.data -= 0.001*b.grad.data
 
  w.grad.zero_()
  b.grad.zero_()
  print("Loss: ", loss.item())

test = mnist.data[41]/255.0
test_flat = test.view(-1, 28*28)
test_pred = simple_nn(test_flat, w, b)
test_pred

test = mnist.data[42]/255.0
test_flat = test.view(-1, 28*28)
test_pred = simple_nn(test_flat, w, b)
test_pred
