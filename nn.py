import torch
from torchvision import datasets
import matplotlib.pyplot as plt
mnist = datasets.MNIST('./data', download=True)

threes = mnist.data[(mnist.targets == 3)]/255.0
sevens = mnist.data[(mnist.targets == 7)]/255.0

len(threes), len(sevens)

def show_image(img):
  plt.imshow(img)
  plt.xticks([])
  plt.yticks([])
  plt.show()
  
show_image(threes[123])
show_image(sevens[78])

print(threes.shape, sevens.shape)

combined_data = torch.cat([threes, sevens])
combined_data.shape

flat_imgs = combined_data.view((-1, 28*28))
flat_imgs.shape

target = torch.tensor([1]*len(threes)+[0]*len(sevens))
target.shape

def sigmoid(x): return 1/(1+torch.exp(-x))

def simple_nn(data, weights, bias): return sigmoid((data@weights) + bias)

def error(pred, target): return ((pred-target)**2).mean()

w = torch.randn((flat_imgs.shape[1], 1), requires_grad=True)
b = torch.randn((1, 1), requires_grad=True)

for i in range(2000):
  pred = simple_nn(flat_imgs, w, b)
  loss = error(pred, target.unsqueeze(1))
  loss.backward()

  w.data -= 0.001*w.grad.data
  b.data -= 0.001*b.grad.data
 
  w.grad.zero_()
  b.grad.zero_()
  
print("Loss: ", loss.item())