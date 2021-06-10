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


def process_heatmap(heatmap_resized):
    """
    Takes in the raw heatmap from the gradcam code, and turns it into a heatmap with HSV color scheme, and uint8
    :param heatmap_raw: raw heatmap resized to fit whatever image you are planning to overlay on top of
    :return: processed heatmap
    """
    # remap to 255 and change to uint8
    heatmap_resize_8uc1 = (heatmap_resized * 255).astype("uint8")
    heatmap_procesed = cv2.applyColorMap(heatmap_resize_8uc1, cv2.COLORMAP_HSV)
    return heatmap_procesed


def apply_heatmap(img, heatmap, alpha=0.4):
    """

    :param img: image in form of numpy array uint8, channel size 3
    :param heatmap: heatmap in form of numpy array uint8, channel size 3
    :return: image that combines the original image with the heatmap
    """
    beta = 1 - alpha
    dst = cv2.addWeighted(img, alpha, heatmap, beta, 0.0)
    return dst


def tensor_to_np_img(img_tensor):
    img = img_tensor.cpu().permute(0, 2, 3, 1).numpy()
    return img


def run(model, dataloader, target_layer):
    font = cv2.FONT_HERSHEY_SIMPLEX
    red = (255, 255, 255)
    blue = (255, 255, 255)
    gradcam = GradCAM(classifier=model, target_layer=target_layer)
    incorrect_indices = []
    for step, data in enumerate(dataloader[0:100]):
        images, labels = data
        images_np = (tensor_to_np_img(images) * 255).astype(np.uint8)
        threshold = 0.5
        heatmap_pred, heatmap_gt, pred_labels = gradcam.cam(input_img=images, label=labels)

        for i in range(labels.shape[0]):
            heatmap_pred_resize = cv2.resize(heatmap_pred[i, ...], (images.shape[-2], images.shape[-1]),
                                             interpolation=cv2.INTER_LINEAR)
            heatmap_gt_resize = cv2.resize(heatmap_gt[i, ...], (images.shape[-2], images.shape[-1]),
                                           interpolation=cv2.INTER_LINEAR)

            heatmap_3ch = np.concatenate([heatmap_pred_resize[..., np.newaxis]] * 3, axis=-1)

            _pred_mask = heatmap_3ch < threshold

            # # remap to 255 and change to uint8
            # heatmap_pred_resize_8uc1 = (heatmap_pred_resize * 255).astype("uint8")
            # heatmap_gt_resize_8uc1 = (heatmap_gt_resize * 255).astype("uint8")
            #
            # # change the heatmap to HSV map for visualzation purpose. The resulting map is now 3 channel
            # heatmap_pred_resize_8uc3 = cv2.applyColorMap(heatmap_pred_resize_8uc1, cv2.COLORMAP_HSV)
            # heatmap_gt_resize_8uc3 = cv2.applyColorMap(heatmap_gt_resize_8uc1, cv2.COLORMAP_HSV)

            heatmap_pred_resize_u8c3 = process_heatmap(heatmap_pred_resize)
            heatmap_gt_resize_u8c3 = process_heatmap(heatmap_gt_resize)

            random_noise = np.random.randn(heatmap_pred_resize_u8c3.shape[0], heatmap_gt_resize_u8c3.shape[1], 1) * 255
            random_noise = np.concatenate([random_noise] * 3, axis=-1)

            random_noise = random_noise.astype(np.uint8)

            pred_mask_black = images_np[i, ...] * _pred_mask
            pred_mask_white = images_np[i, ...] * _pred_mask
            pred_mask_white[pred_mask_white == 0] = 255

            pred_mask_noise = images_np[i, ...] * _pred_mask + random_noise * (1 - _pred_mask).astype(np.uint8)

            pred_mask_black = cv2.resize(pred_mask_black, (pred_mask_black.shape[0] * 2, pred_mask_black.shape[1] * 2),
                                         interpolation=cv2.INTER_AREA)

            # TODO: Try getting heatmap of masked image
            # heatmap_mask_pred, heatmap_mask_gt, pred_label = gradcam.cam(input_img=pred_mask_black,
            #                                                              label=labels[[i], ...])

            pred_mask_white = cv2.resize(pred_mask_white, (pred_mask_white.shape[0] * 2, pred_mask_white.shape[1] * 2),
                                         interpolation=cv2.INTER_AREA)

            pred_mask_noise = cv2.resize(pred_mask_noise, (pred_mask_noise.shape[0] * 2, pred_mask_noise.shape[1] * 2),
                                         interpolation=cv2.INTER_AREA)

            curr_img = images_np[i, ...]
            dst_pred = apply_heatmap(img=curr_img, heatmap=heatmap_pred_resize_u8c3)
            dst_gt = apply_heatmap(img=curr_img, heatmap=heatmap_gt_resize_u8c3)

            dst_pred = cv2.resize(dst_pred, (dst_pred.shape[0] * 2, dst_pred.shape[1] * 2),
                                  interpolation=cv2.INTER_AREA)

            dst_pred = cv2.putText(dst_pred, f"Pred:{LABELS[int(pred_labesl[i].item())]}",
                                   org=(0 + 5, 0 + 5), fontFace=font, fontScale=0.25,
                                   color=blue, thickness=1)

            dst_gt = cv2.resize(dst_gt, (dst_gt.shape[0] * 2, dst_gt.shape[1] * 2), interpolation=cv2.INTER_AREA)
            dst_gt = cv2.putText(dst_gt, f"GT:{LABELS[labels[i].item()]}",
                                 org=(0 + 5, 0 + 5), fontFace=font, fontScale=0.25,
                                 color=red, thickness=1)

            orig = cv2.resize(curr_img, (curr_img.shape[0] * 2, curr_img.shape[1] * 2), interpolation=cv2.INTER_AREA)

            dst = np.concatenate([orig, dst_pred, pred_mask_black, pred_mask_white, pred_mask_noise, dst_gt], axis=1)

            filename = f"outputs/{step}_pred_{LABELS[int(pred_labels[i].item())]}_gt_{LABELS[labels[i].item()]}.jpg"

            cv2.imwrite(filename, cv2.cvtColor(dst, cv2.COLOR_RGB2BGR))

            if pred_labels != labels[i]:
                incorrect_indices.append(i)


if __name__ == '__main__':
    cam_model = load_pretrained_model(path="cifar100_resnet20.pt")
    target_layer = cam_model.layer3[-1]
    dataloader = get_dataloader(isTrain=False)

    print("Begin Program")
    run(model=cam_model, dataloader=dataloader, target_layer=target_layer)
