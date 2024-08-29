import cv2
import os
import torch

SIZE = (512, 512)
PATH_IMAGES = "training_images/initial_images/"
PATH_COLOR_IMAGES = "training_images/preprocessed_images/"
PATH_GRAY_IMAGES = "training_images/gray_scale_images/"
PATH_GRAY_EVALUATION_IMAGES = "evaluation_images/gray_scale_images/"
PATH_COLOR_EVALUATION_IMAGES = "evaluation_images/color_images/"

def _new_color_img_path(i : int):
    return PATH_COLOR_IMAGES + "color_image" + str(i) + ".jpg"

def _new_gray_img_path(i : int):
    return PATH_GRAY_IMAGES + "gray_image" + str(i) + ".jpg"

def return_gray_image(i : int):
    img =  cv2.imread(_new_gray_img_path(i), cv2.IMREAD_GRAYSCALE)
    img = torch.Tensor(img)
    img = img.unsqueeze(0)
    img = img.unsqueeze(0)
    return img

def return_color_image(i : int):
    return cv2.imread(_new_color_img_path(i), cv2.IMREAD_COLOR)

def return_lab_image(i : int):
    img_lab = cv2.cvtColor(return_color_image(i), cv2.COLOR_RGB2Lab) # image could differ here
    L, a, b = cv2.split(img_lab)
    L = torch.Tensor(L)
    L = L.unsqueeze(0)
    L = L.unsqueeze(0)
    a = torch.Tensor(a)
    b = torch.Tensor(b)
    return L, torch.stack([a, b], dim=0)

def _get_current_img_path(idx : int):
    value = "000" + str(idx) + ".png"
    return PATH_IMAGES + value

def _get_new_name(idx : int):
    return PATH_IMAGES + "color_image" + str(idx) + ".jpg"

def _rename_multiple_image(start_idx : int, end_idx : int):
    for i in range(start_idx, end_idx, 1):
        os.rename(_get_new_name(i), _get_new_name(i - 532))

def _compute__grayscale_from_color(start_idx : int, end_idx : int):
    for i in range(start_idx, end_idx, 1):
        image_path = _new_color_img_path(i)
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        cv2.imwrite(_new_gray_img_path(i), gray_image)
