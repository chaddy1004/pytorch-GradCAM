import torchvision
from grad_cam import GradCAM
from model import load_pretrained_model
from dataset import get_dataloader
import cv2
import numpy as np

LABELS = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl',
          'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee',
          'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant',
          'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower',
          'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom',
          'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate',
          'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark',
          'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower',
          'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
          'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']


def tensor_to_np_img(img_tensor):
    img = img_tensor.cpu().permute(0, 2, 3, 1).numpy()
    return img


def run(model, dataloader, target_layer):
    font = cv2.FONT_HERSHEY_SIMPLEX
    colour = (255, 0, 0)
    gradcam = GradCAM(classifier=model, target_layer=target_layer)
    for step, data in enumerate(dataloader):
        images, labels = data
        images_np = (tensor_to_np_img(images) * 255).astype(np.uint8)

        heatmap_pred, heatmap_gt, pred_label = gradcam.cam(input_img=images, label=labels)
        for i in range(labels.shape[0]):
            heatmap_pred_resize = cv2.resize(heatmap_pred[i, ...], (images.shape[-2], images.shape[-1]),
                                             interpolation=cv2.INTER_LINEAR)
            heatmap_gt_resize = cv2.resize(heatmap_gt[i, ...], (images.shape[-2], images.shape[-1]),
                                           interpolation=cv2.INTER_LINEAR)

            heatmap_pred_resize_8uc1 = (heatmap_pred_resize * 255).astype("uint8")
            heatmap_gt_resize_8uc1 = (heatmap_gt_resize * 255).astype("uint8")

            heatmap_pred_resize_8uc1 = cv2.applyColorMap(heatmap_pred_resize_8uc1, cv2.COLORMAP_HSV)
            heatmap_gt_resize_8uc1 = cv2.applyColorMap(heatmap_gt_resize_8uc1, cv2.COLORMAP_HSV)

            alpha = 0.4
            beta = 1 - alpha
            orig = images_np[i, ...]
            dst_pred = cv2.addWeighted(images_np[i, ...], alpha, heatmap_pred_resize_8uc1, beta, 0.0)
            dst_gt = cv2.addWeighted(images_np[i, ...], alpha, heatmap_gt_resize_8uc1, beta, 0.0)

            dst_pred = cv2.resize(dst_pred, (dst_pred.shape[0] * 2, dst_pred.shape[1] * 2),
                                  interpolation=cv2.INTER_AREA)
            dst_gt = cv2.resize(dst_gt, (dst_gt.shape[0] * 2, dst_gt.shape[1] * 2), interpolation=cv2.INTER_AREA)
            orig = cv2.resize(orig, (orig.shape[0] * 2, orig.shape[1] * 2), interpolation=cv2.INTER_AREA)

            dst = np.concatenate([orig, dst_pred, dst_gt], axis=1)

            # dst = cv2.putText(dst, "Pred",
            #                   org=(0 + 30, 0 + 30), fontFace=font, fontScale=1,
            #                   color=colour, thickness=2)
            #
            # dst = cv2.putText(dst, "GT",
            #                   org=(dst.shape[-1]//2 + 30, 0 + 30), fontFace=font, fontScale=1,
            #                   color=colour, thickness=2)

            filename = f"outputs/{step}_pred_{LABELS[int(pred_label[i].item())]}_gt_{LABELS[labels[i].item()]}.jpg"

            cv2.imwrite(filename, cv2.cvtColor(dst, cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    cam_model = load_pretrained_model(path="cifar100_resnet20.pt")
    target_layer = cam_model.layer3[-1]
    dataloader = get_dataloader(isTrain=False)

    print("Begin Program")
    run(model=cam_model, dataloader=dataloader, target_layer=target_layer)
