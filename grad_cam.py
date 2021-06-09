import torch
import torch.nn.functional as F
import numpy as np


class GradCAM:
    def __init__(self, classifier):
        self.classifier = classifier if not classifier.training else classifier.eval()
        self.encoded = None
        self.pred = None
        self.top_pred_index = 0
        self.ch = 0

    def forward(self, img):
        self.pred, self.encoded = self.classifier(img, mode="cam")
        _, self.ch, *_ = self.encoded.size()
        print(self.pred, self.encoded.size())

    def backward(self):
        grad = torch.autograd.grad(outputs=self.pred, inputs=self.encoded, grad_outputs=torch.ones_like(self.pred))[0]
        # only leaving the channel
        pooled_grads = torch.mean(grad, [-2, -1])
        # reshaping to do channel-wise element-wise multiplication
        # going from (batch, ch) to (batch, ch, 1, 1)
        pooled_grads = pooled_grads.view(-1, self.ch, 1, 1)
        self.heatmap_tensor = pooled_grads * self.encoded
        self.heatmap_tensor = torch.mean(self.heatmap_tensor, dim=1)
        # to have mimimum value as 0
        self.heatmap_tensor = F.relu(self.heatmap_tensor)
        # normalizing between 0-1
        self.heatmap_tensor = self.heatmap_tensor / torch.max(self.heatmap_tensor)
        return np.squeeze(self.heatmap_tensor.detach().numpy())

    def cam(self, input_img):
        self.forward(img=input_img)
        return self.backward()
