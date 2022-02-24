# Import Dependencies
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import models
from skimage.io import imread
from skimage.transform import resize
import cv2


class GradCamModel(nn.Module):
    def __init__(self, model, last_layer):
        super().__init__()
        self.gradients = None
        self.tensorhook = []
        self.layerhook = []
        self.selected_out = None
        self.model = model
        self.last_layer = last_layer

        """ 
        The gradients of the output with respect to the activations are merely intermediate values
        They are discarded as soon as the gradient propagates through them on the way back
        Attaching hook to intermediate values will pull the gradients out of the model before they are discarded
        
        1. forward_hook() : The hook will be called every time after forward() has computed an output
        2. register_module_forward_hook() : Registers a global forward hook for all the modules

        layerhook will store all the foward pass hooks
        """
        self.layerhook.append(self.model.last_layer.register_forward_hook(self.forward_hook()))

        for p in self.model.parameters():
            p.requires_grad = True

    def activations_hook(self, grad):
        # grad: The value contained in the grad attribute of the tensor after backward is called
        self.gradients = grad

    def get_activation_gradients(self):
        return self.gradients

    # Hooks for keeping track of the gradients
    def forward_hook(self):
        def hook(module, inp, out):
            self.selected_out = out
            self.tensorhook.append(out.register_hook(self.activations_hook))

        return hook

    def forward(self, x):
        # forward pass of the model to get predictions
        out = self.model(x)
        # returning the output of the model and activation gradients
        return out, self.selected_out


def compute_heatmap(input_image):

    gcmodel = GradCamModel().to("cuda:0")

    """
  Give INPUT IMAGES to gradCAM model
  model_output  : Predicted class in the image
  activations   : Selected Region in the image
  """
    model_output, activations = gcmodel(input_image)

    """
  detach() : Returns a new Tensor, detached from the current graph
  """
    activations = activations.detach().cpu()

    # array[600] need to change!
    loss = nn.CrossEntropyLoss()(model_output, torch.from_numpy(np.array([600])).to("cuda:0"))

    # Backpropogation
    loss.backward()

    # Calculating gradients
    grads = gcmodel.get_activation_gradients().detach().cpu()

    # Calculating activations using pooled gradients
    pooled_grads = torch.mean(grads, dim=[0, 2, 3]).detach().cpu()
    for i in range(activations.shape[1]):
        activations[:, i, :, :] += pooled_grads[i]

    # Create Heatmap
    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap_max = heatmap.max(axis=0)[0]
    heatmap /= heatmap_max

    # Display Heatmap
    heatmap = cv2.resize(heatmap, (input_image.shape[1], input_image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    intensity = 0.5
    superimposed_img = heatmap * intensity + input_image

    plt.imshow(superimposed_img)
