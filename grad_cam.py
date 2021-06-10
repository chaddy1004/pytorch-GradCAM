import torch
import torch.nn.functional as F
import numpy as np


class GradCAM:
    def __init__(self, classifier, target_layer):
        self.classifier = classifier if not classifier.training else classifier.eval()
        target_layer.register_forward_hook(self.save_feature)
        target_layer.register_backward_hook(self.save_grad)
        self.encoded = None
        self.pred = None
        self.top_pred_index = 0
        self.ch = 0
        self.feature = None
        self.gradient = None

    def save_feature(self, module, input, output):
        self.feature = output.cpu().detach()

    def save_grad(self, module, grad_input, grad_output):
        self.gradient = grad_output[0].cpu().detach()

    def forward(self, img):
        self.pred, self.encoded = self.classifier(img, cam=True)
        print(torch.all(torch.eq(self.encoded, self.feature)))
        _, self.ch, *_ = self.encoded.size()

    def backward(self, label):
        self.classifier.zero_grad()
        label = label.long()
        loss = self.pred[:, label]
        loss.backward(retain_graph=True)

        # self.classifier.zero_grad()
        #
        # loss = self.pred[:, label]
        #
        # grad = torch.autograd.grad(outputs=loss, inputs=self.encoded, retain_graph=True)[0]
        # print(torch.all(torch.eq(grad, self.gradient)))
        # only leaving the channel
        print(f"gradient shape: {self.gradient.shape}")

        pooled_grads = torch.mean(self.gradient, [-2, -1])
        print(f"averaged gradient shaoe: {pooled_grads.shape}")

        # reshaping to do channel-wise element-wise multiplication
        # going from (batch, ch) to (batch, ch, 1, 1)
        pooled_grads = pooled_grads.view(-1, self.ch, 1, 1)
        self.heatmap_tensor = pooled_grads * self.encoded
        self.heatmap_tensor = torch.mean(self.heatmap_tensor, dim=1)
        # to have mimimum value as 0
        self.heatmap_tensor = F.relu(self.heatmap_tensor)
        # normalizing between 0-1
        self.heatmap_tensor = self.heatmap_tensor / torch.max(self.heatmap_tensor)

        print(f"heatmap shape: {self.heatmap_tensor.shape}")
        return self.heatmap_tensor.detach().numpy()

    def cam(self, input_img, label):
        self.forward(img=input_img)
        # pred_label = self.pred >= torch.max(self.pred)
        pred_label = torch.argmax(self.pred, dim=-1)
        pred_label = pred_label.float()

        print(f"pred_label: {pred_label}, gt_label: {label}")

        pred_label.requires_grad = True
        label = label.float()
        label.requires_grad = True

        # since I am using one forward for both backwards, I need to set retain_graph = True
        return self.backward(pred_label), self.backward(label), pred_label
