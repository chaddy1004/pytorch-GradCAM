import torchvision
from grad_cam import GradCAM
from model import load_pretrained_model
from dataset import get_dataloader
import cv2
import numpy as np
import torch
import os

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


def np_img_to_tensor(img, normalize):
    if normalize:
        img = img / np.max(img)
        img = img.astype(np.float)
    img_tensor = torch.from_numpy(img)
    img_tensor = img_tensor.permute(2, 0, 1)
    img_tensor = torch.unsqueeze(img_tensor, 0)
    return img_tensor


def run(model, dataloader, target_layer, num_data=30):
    if not os.path.isdir("incorrects"):
        os.mkdir("incorrects")

    if not os.path.isdir("corrects"):
        os.mkdir("corrects")

    if not os.path.isdir("outputs"):
        os.mkdir("outputs")

    img_scale = 3

    font = cv2.FONT_HERSHEY_SIMPLEX
    red = (255, 255, 255)
    blue = (255, 255, 255)
    gradcam = GradCAM(classifier=model, target_layer=target_layer)
    incorrect_labels = []
    incorrect_pred_heatmaps = []
    incorrect_gt_heatmaps = []
    incorrect_orig_imgs = []
    incorrect_orig_imgs_tensor = []
    incorrect_black_masks = []
    incorrect_white_masks = []
    incorrect_noisy_masks = []

    correct_labels = []
    correct_pred_heatmaps = []
    correct_gt_heatmaps = []
    correct_orig_imgs = []
    correct_orig_imgs_tensor = []
    correct_black_masks = []
    correct_white_masks = []
    correct_noisy_masks = []

    for step, data in enumerate(dataloader):
        print(step, num_data)
        if step == num_data:
            break
        images, labels = data
        images_np = (tensor_to_np_img(images) * 255).astype(np.uint8)
        threshold = 0.5

        heatmap_pred, heatmap_gt, pred_labels = gradcam.cam(input_img=images, label=labels)

        for i in range(labels.shape[0]):
            heatmap_pred_resize = cv2.resize(heatmap_pred[i, ...], (images.shape[-2], images.shape[-1]),
                                             interpolation=cv2.INTER_LINEAR)
            heatmap_gt_resize = cv2.resize(heatmap_gt[i, ...], (images.shape[-2], images.shape[-1]),
                                           interpolation=cv2.INTER_LINEAR)

            # get the map based on the ground truth
            heatmap_3ch = np.concatenate([heatmap_gt_resize[..., np.newaxis]] * 3, axis=-1)

            _pred_mask = heatmap_3ch < threshold

            heatmap_pred_resize_u8c3 = process_heatmap(heatmap_pred_resize)
            heatmap_gt_resize_u8c3 = process_heatmap(heatmap_gt_resize)

            random_noise = np.random.randn(heatmap_pred_resize_u8c3.shape[0], heatmap_gt_resize_u8c3.shape[1], 1) * 255
            random_noise = np.concatenate([random_noise] * 3, axis=-1)

            random_noise = random_noise.astype(np.uint8)
            # black mask
            pred_mask_black_32 = images_np[i, ...] * _pred_mask
            pred_mask_black = cv2.resize(pred_mask_black_32,
                                         (pred_mask_black_32.shape[0] * img_scale,
                                          pred_mask_black_32.shape[1] * img_scale),
                                         interpolation=cv2.INTER_AREA)

            # white mask
            pred_mask_white_32 = images_np[i, ...] * _pred_mask
            pred_mask_white_32[pred_mask_white_32 == 0] = 255
            pred_mask_white = cv2.resize(pred_mask_white_32,
                                         (pred_mask_white_32.shape[0] * img_scale,
                                          pred_mask_white_32.shape[1] * img_scale),
                                         interpolation=cv2.INTER_AREA)

            # noisy mask
            pred_mask_noise_32 = images_np[i, ...] * _pred_mask + random_noise * (1 - _pred_mask).astype(np.uint8)

            pred_mask_noise = cv2.resize(pred_mask_noise_32,
                                         (pred_mask_noise_32.shape[0] * img_scale,
                                          pred_mask_noise_32.shape[1] * img_scale),
                                         interpolation=cv2.INTER_AREA)

            curr_img = images_np[i, ...]
            dst_pred = apply_heatmap(img=curr_img, heatmap=heatmap_pred_resize_u8c3)
            dst_gt = apply_heatmap(img=curr_img, heatmap=heatmap_gt_resize_u8c3)

            dst_pred = cv2.resize(dst_pred, (dst_pred.shape[0] * img_scale, dst_pred.shape[1] * img_scale),
                                  interpolation=cv2.INTER_AREA)

            dst_pred = cv2.putText(dst_pred, f"{LABELS[int(pred_labels[i].item())]}",
                                   org=(0 + 5, 0 + 15), fontFace=font, fontScale=0.5,
                                   color=blue, thickness=1)

            dst_gt = cv2.resize(dst_gt, (dst_gt.shape[0] * img_scale, dst_gt.shape[1] * img_scale),
                                interpolation=cv2.INTER_AREA)
            dst_gt = cv2.putText(dst_gt, f"GT: {LABELS[labels[i].item()]}",
                                 org=(0 + 5, 0 + 15), fontFace=font, fontScale=0.5,
                                 color=red, thickness=1)

            orig = cv2.resize(curr_img, (curr_img.shape[0] * img_scale, curr_img.shape[1] * img_scale),
                              interpolation=cv2.INTER_AREA)

            dst = np.concatenate([orig, dst_pred, pred_mask_black, pred_mask_white, pred_mask_noise, dst_gt], axis=1)

            filename = f"outputs/{step}_pred_{LABELS[int(pred_labels[i].item())]}_gt_{LABELS[labels[i].item()]}.jpg"

            cv2.imwrite(filename, cv2.cvtColor(dst, cv2.COLOR_RGB2BGR))

            if pred_labels != labels[i]:
                incorrect_labels.append(labels[i])
                incorrect_orig_imgs_tensor.append(images)
                incorrect_orig_imgs.append(curr_img)
                incorrect_pred_heatmaps.append(dst_pred)
                incorrect_gt_heatmaps.append(dst_gt)
                incorrect_black_masks.append(pred_mask_black_32)
                incorrect_white_masks.append(pred_mask_white_32)
                incorrect_noisy_masks.append(pred_mask_noise_32)

            else:
                correct_labels.append(labels[i])
                correct_orig_imgs_tensor.append(images)
                correct_orig_imgs.append(curr_img)
                correct_pred_heatmaps.append(dst_pred)
                correct_gt_heatmaps.append(dst_gt)
                correct_black_masks.append(pred_mask_black_32)
                correct_white_masks.append(pred_mask_white_32)
                correct_noisy_masks.append(pred_mask_noise_32)

    for i, img in enumerate(incorrect_orig_imgs):
        label = incorrect_labels[i]
        orig_pred_heatmap = incorrect_pred_heatmaps[i]
        orig_gt_heatmap = incorrect_gt_heatmaps[i]
        black_mask = incorrect_black_masks[i]
        black_mask_tensor = np_img_to_tensor(black_mask, normalize=True).float()

        white_mask = incorrect_white_masks[i]
        white_mask_tensor = np_img_to_tensor(white_mask, normalize=True).float()

        noisy_mask = incorrect_noisy_masks[i]
        noisy_mask_tensor = np_img_to_tensor(noisy_mask, normalize=True).float()

        heatmap_pred_black, heatmap_gt_black, pred_label_black = gradcam.cam(input_img=black_mask_tensor,
                                                                             label=label)

        heatmap_pred_white, heatmap_gt_white, pred_label_white = gradcam.cam(input_img=white_mask_tensor,
                                                                             label=label)

        heatmap_pred_noisy, heatmap_gt_noisy, pred_label_noisy = gradcam.cam(input_img=noisy_mask_tensor,
                                                                             label=label)

        heatmap_pred_black = cv2.resize(heatmap_pred_black[0, ...], (img.shape[0], img.shape[1]),
                                        interpolation=cv2.INTER_LINEAR)
        heatmap_pred_white = cv2.resize(heatmap_pred_white[0, ...], (img.shape[0], img.shape[1]),
                                        interpolation=cv2.INTER_LINEAR)
        heatmap_pred_noisy = cv2.resize(heatmap_pred_noisy[0, ...], (img.shape[0], img.shape[1]),
                                        interpolation=cv2.INTER_LINEAR)
        heatmap_gt_black = cv2.resize(heatmap_gt_black[0, ...], (img.shape[0], img.shape[1]),
                                      interpolation=cv2.INTER_LINEAR)
        heatmap_gt_white = cv2.resize(heatmap_gt_white[0, ...], (img.shape[0], img.shape[1]),
                                      interpolation=cv2.INTER_LINEAR)
        heatmap_gt_noisy = cv2.resize(heatmap_gt_noisy[0, ...], (img.shape[0], img.shape[1]),
                                      interpolation=cv2.INTER_LINEAR)

        heatmap_pred_black_8uc1 = process_heatmap(heatmap_pred_black)
        heatmap_pred_white_8uc1 = process_heatmap(heatmap_pred_white)
        heatmap_pred_noisy_8uc1 = process_heatmap(heatmap_pred_noisy)
        heatmap_gt_black_8uc1 = process_heatmap(heatmap_gt_black)
        heatmap_gt_white_8uc1 = process_heatmap(heatmap_gt_white)
        heatmap_gt_noisy_8uc1 = process_heatmap(heatmap_gt_noisy)

        dst_black_pred = apply_heatmap(black_mask, heatmap_pred_black_8uc1)
        dst_white_pred = apply_heatmap(white_mask, heatmap_pred_white_8uc1)
        dst_noisy_pred = apply_heatmap(noisy_mask, heatmap_pred_noisy_8uc1)
        dst_black_gt = apply_heatmap(black_mask, heatmap_gt_black_8uc1)
        dst_white_gt = apply_heatmap(white_mask, heatmap_gt_white_8uc1)
        dst_noisy_gt = apply_heatmap(noisy_mask, heatmap_gt_noisy_8uc1)

        dst_black_pred = cv2.resize(dst_black_pred,
                                    (dst_black_pred.shape[0] * img_scale, dst_black_pred.shape[1] * img_scale),
                                    interpolation=cv2.INTER_AREA)

        dst_black_pred = cv2.putText(dst_black_pred, f"{LABELS[int(pred_label_black.item())]}",
                                     org=(0 + 5, 0 + 15), fontFace=font, fontScale=0.5,
                                     color=blue, thickness=1)

        dst_white_pred = cv2.resize(dst_white_pred,
                                    (dst_white_pred.shape[0] * img_scale, dst_white_pred.shape[1] * img_scale),
                                    interpolation=cv2.INTER_AREA)

        dst_white_pred = cv2.putText(dst_white_pred, f"{LABELS[int(pred_label_white.item())]}",
                                     org=(0 + 5, 0 + 15), fontFace=font, fontScale=0.5,
                                     color=blue, thickness=1)

        dst_noisy_pred = cv2.resize(dst_noisy_pred,
                                    (dst_noisy_pred.shape[0] * img_scale, dst_noisy_pred.shape[1] * img_scale),
                                    interpolation=cv2.INTER_AREA)

        dst_noisy_pred = cv2.putText(dst_noisy_pred, f"{LABELS[int(pred_label_noisy.item())]}",
                                     org=(0 + 5, 0 + 15), fontFace=font, fontScale=0.5,
                                     color=blue, thickness=1)

        dst_black_gt = cv2.resize(dst_black_gt,
                                  (dst_black_gt.shape[0] * img_scale, dst_black_gt.shape[1] * img_scale),
                                  interpolation=cv2.INTER_AREA)

        dst_black_gt = cv2.putText(dst_black_gt, f"{LABELS[int(label.item())]}",
                                   org=(0 + 5, 0 + 15), fontFace=font, fontScale=0.5,
                                   color=blue, thickness=1)

        dst_white_gt = cv2.resize(dst_white_gt,
                                  (dst_white_gt.shape[0] * img_scale, dst_white_gt.shape[1] * img_scale),
                                  interpolation=cv2.INTER_AREA)

        dst_white_gt = cv2.putText(dst_white_gt, f"{LABELS[int(label.item())]}",
                                   org=(0 + 5, 0 + 15), fontFace=font, fontScale=0.5,
                                   color=blue, thickness=1)

        dst_noisy_gt = cv2.resize(dst_noisy_gt,
                                  (dst_noisy_gt.shape[0] * img_scale, dst_noisy_gt.shape[1] * img_scale),
                                  interpolation=cv2.INTER_AREA)

        dst_noisy_gt = cv2.putText(dst_noisy_gt, f"{LABELS[int(label.item())]}",
                                   org=(0 + 5, 0 + 15), fontFace=font, fontScale=0.5,
                                   color=blue, thickness=1)

        black_mask = cv2.resize(black_mask,
                                (black_mask.shape[0] * img_scale, black_mask.shape[1] * img_scale),
                                interpolation=cv2.INTER_AREA)
        white_mask = cv2.resize(white_mask,
                                (white_mask.shape[0] * img_scale, white_mask.shape[1] * img_scale),
                                interpolation=cv2.INTER_AREA)
        noisy_mask = cv2.resize(noisy_mask,
                                (noisy_mask.shape[0] * img_scale, noisy_mask.shape[1] * img_scale),
                                interpolation=cv2.INTER_AREA)

        img = cv2.resize(img, (img.shape[0] * img_scale, img.shape[1] * img_scale), interpolation=cv2.INTER_AREA)

        img = cv2.putText(img, f"GT: {LABELS[int(label.item())]}",
                          org=(0 + 5, 0 + 15), fontFace=font, fontScale=0.5,
                          color=blue, thickness=1)

        texts = ["GT Image", "GT Map", "Pred Map", "Mask", "PM Mask", "GM Mask"]
        text_row = np.ones((img.shape[0] // 4, img.shape[0] * len(texts), 3)) * 255

        for i, text in enumerate(texts):
            text_row = cv2.putText(text_row, text,
                                   org=(i * img.shape[0] + 15, 15), fontFace=font, fontScale=0.5,
                                   color=(0, 0, 0), thickness=1)

        black_row = np.concatenate([img, orig_gt_heatmap, orig_pred_heatmap, black_mask, dst_black_pred, dst_black_gt],
                                   axis=1)
        white_row = np.concatenate([img, orig_gt_heatmap, orig_pred_heatmap, white_mask, dst_white_pred, dst_white_gt],
                                   axis=1)
        noisy_row = np.concatenate([img, orig_gt_heatmap, orig_pred_heatmap, noisy_mask, dst_noisy_pred, dst_noisy_gt],
                                   axis=1)

        filename = f"incorrects/{i}_gt_{LABELS[label.item()]}.jpg"

        total_img = np.concatenate([text_row.astype(np.uint8), black_row, white_row, noisy_row], axis=0)

        cv2.imwrite(filename, cv2.cvtColor(total_img, cv2.COLOR_RGB2BGR))

    for i, img in enumerate(correct_orig_imgs):
        label = correct_labels[i]
        orig_pred_heatmap = correct_pred_heatmaps[i]
        orig_gt_heatmap = correct_gt_heatmaps[i]

        black_mask = correct_black_masks[i]
        black_mask_tensor = np_img_to_tensor(black_mask, normalize=True).float()

        white_mask = correct_white_masks[i]
        white_mask_tensor = np_img_to_tensor(white_mask, normalize=True).float()

        noisy_mask = correct_noisy_masks[i]
        noisy_mask_tensor = np_img_to_tensor(noisy_mask, normalize=True).float()

        heatmap_pred_black, heatmap_gt_black, pred_label_black = gradcam.cam(input_img=black_mask_tensor,
                                                                             label=label)

        heatmap_pred_white, heatmap_gt_white, pred_label_white = gradcam.cam(input_img=white_mask_tensor,
                                                                             label=label)

        heatmap_pred_noisy, heatmap_gt_noisy, pred_label_noisy = gradcam.cam(input_img=noisy_mask_tensor,
                                                                             label=label)

        heatmap_pred_black = cv2.resize(heatmap_pred_black[0, ...], (img.shape[0], img.shape[1]),
                                        interpolation=cv2.INTER_LINEAR)
        heatmap_pred_white = cv2.resize(heatmap_pred_white[0, ...], (img.shape[0], img.shape[1]),
                                        interpolation=cv2.INTER_LINEAR)
        heatmap_pred_noisy = cv2.resize(heatmap_pred_noisy[0, ...], (img.shape[0], img.shape[1]),
                                        interpolation=cv2.INTER_LINEAR)
        heatmap_gt_black = cv2.resize(heatmap_gt_black[0, ...], (img.shape[0], img.shape[1]),
                                      interpolation=cv2.INTER_LINEAR)
        heatmap_gt_white = cv2.resize(heatmap_gt_white[0, ...], (img.shape[0], img.shape[1]),
                                      interpolation=cv2.INTER_LINEAR)
        heatmap_gt_noisy = cv2.resize(heatmap_gt_noisy[0, ...], (img.shape[0], img.shape[1]),
                                      interpolation=cv2.INTER_LINEAR)

        heatmap_pred_black_8uc1 = process_heatmap(heatmap_pred_black)
        heatmap_pred_white_8uc1 = process_heatmap(heatmap_pred_white)
        heatmap_pred_noisy_8uc1 = process_heatmap(heatmap_pred_noisy)
        heatmap_gt_black_8uc1 = process_heatmap(heatmap_gt_black)
        heatmap_gt_white_8uc1 = process_heatmap(heatmap_gt_white)
        heatmap_gt_noisy_8uc1 = process_heatmap(heatmap_gt_noisy)

        dst_black_pred = apply_heatmap(black_mask, heatmap_pred_black_8uc1)
        dst_white_pred = apply_heatmap(white_mask, heatmap_pred_white_8uc1)
        dst_noisy_pred = apply_heatmap(noisy_mask, heatmap_pred_noisy_8uc1)
        dst_black_gt = apply_heatmap(black_mask, heatmap_gt_black_8uc1)
        dst_white_gt = apply_heatmap(white_mask, heatmap_gt_white_8uc1)
        dst_noisy_gt = apply_heatmap(noisy_mask, heatmap_gt_noisy_8uc1)

        dst_black_pred = cv2.resize(dst_black_pred,
                                    (dst_black_pred.shape[0] * img_scale, dst_black_pred.shape[1] * img_scale),
                                    interpolation=cv2.INTER_AREA)

        dst_black_pred = cv2.putText(dst_black_pred, f"{LABELS[int(pred_label_black.item())]}",
                                     org=(0 + 5, 0 + 15), fontFace=font, fontScale=0.5,
                                     color=blue, thickness=1)

        dst_white_pred = cv2.resize(dst_white_pred,
                                    (dst_white_pred.shape[0] * img_scale, dst_white_pred.shape[1] * img_scale),
                                    interpolation=cv2.INTER_AREA)

        dst_white_pred = cv2.putText(dst_white_pred, f"{LABELS[int(pred_label_white.item())]}",
                                     org=(0 + 5, 0 + 15), fontFace=font, fontScale=0.5,
                                     color=blue, thickness=1)

        dst_noisy_pred = cv2.resize(dst_noisy_pred,
                                    (dst_noisy_pred.shape[0] * img_scale, dst_noisy_pred.shape[1] * img_scale),
                                    interpolation=cv2.INTER_AREA)

        dst_noisy_pred = cv2.putText(dst_noisy_pred, f"{LABELS[int(pred_label_noisy.item())]}",
                                     org=(0 + 5, 0 + 15), fontFace=font, fontScale=0.5,
                                     color=blue, thickness=1)

        dst_black_gt = cv2.resize(dst_black_gt,
                                  (dst_black_gt.shape[0] * img_scale, dst_black_gt.shape[1] * img_scale),
                                  interpolation=cv2.INTER_AREA)

        dst_black_gt = cv2.putText(dst_black_gt, f"{LABELS[int(label.item())]}",
                                   org=(0 + 5, 0 + 15), fontFace=font, fontScale=0.5,
                                   color=blue, thickness=1)

        dst_white_gt = cv2.resize(dst_white_gt,
                                  (dst_white_gt.shape[0] * img_scale, dst_white_gt.shape[1] * img_scale),
                                  interpolation=cv2.INTER_AREA)

        dst_white_gt = cv2.putText(dst_white_gt, f"{LABELS[int(label.item())]}",
                                   org=(0 + 5, 0 + 15), fontFace=font, fontScale=0.5,
                                   color=blue, thickness=1)

        dst_noisy_gt = cv2.resize(dst_noisy_gt,
                                  (dst_noisy_gt.shape[0] * img_scale, dst_noisy_gt.shape[1] * img_scale),
                                  interpolation=cv2.INTER_AREA)

        dst_noisy_gt = cv2.putText(dst_noisy_gt, f"{LABELS[int(label.item())]}",
                                   org=(0 + 5, 0 + 15), fontFace=font, fontScale=0.5,
                                   color=blue, thickness=1)

        black_mask = cv2.resize(black_mask,
                                (black_mask.shape[0] * img_scale, black_mask.shape[1] * img_scale),
                                interpolation=cv2.INTER_AREA)
        white_mask = cv2.resize(white_mask,
                                (white_mask.shape[0] * img_scale, white_mask.shape[1] * img_scale),
                                interpolation=cv2.INTER_AREA)
        noisy_mask = cv2.resize(noisy_mask,
                                (noisy_mask.shape[0] * img_scale, noisy_mask.shape[1] * img_scale),
                                interpolation=cv2.INTER_AREA)

        img = cv2.resize(img, (img.shape[0] * img_scale, img.shape[1] * img_scale), interpolation=cv2.INTER_AREA)

        img = cv2.putText(img, f"GT: {LABELS[int(label.item())]}",
                          org=(0 + 5, 0 + 15), fontFace=font, fontScale=0.5,
                          color=blue, thickness=1)

        texts = ["GT Image", "GT Map", "Pred Map", "Mask", "PM Mask", "GM Mask"]
        text_row = np.ones((img.shape[0] // 4, img.shape[0] * len(texts), 3)) * 255

        for i, text in enumerate(texts):
            text_row = cv2.putText(text_row, text,
                                   org=(i * img.shape[0] + 15, 15), fontFace=font, fontScale=0.5,
                                   color=(0, 0, 0), thickness=1)

        black_row = np.concatenate([img, orig_gt_heatmap, orig_pred_heatmap, black_mask, dst_black_pred, dst_black_gt],
                                   axis=1)
        white_row = np.concatenate([img, orig_gt_heatmap, orig_pred_heatmap, white_mask, dst_white_pred, dst_white_gt],
                                   axis=1)
        noisy_row = np.concatenate([img, orig_gt_heatmap, orig_pred_heatmap, noisy_mask, dst_noisy_pred, dst_noisy_gt],
                                   axis=1)

        filename = f"corrects/{i}_gt_{LABELS[label.item()]}.jpg"

        total_img = np.concatenate([text_row.astype(np.uint8), black_row, white_row, noisy_row], axis=0)

        cv2.imwrite(filename, cv2.cvtColor(total_img, cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    cam_model = load_pretrained_model(path="cifar100_resnet20.pt")
    target_layer = cam_model.layer3[-1]
    dataloader = get_dataloader(isTrain=False)

    print("Begin Program")
    run(model=cam_model, dataloader=dataloader, target_layer=target_layer)
