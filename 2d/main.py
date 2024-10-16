import torch
from segment_anything import sam_model_registry
import cv2
from segment_anything import SamPredictor
import numpy as np

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"

sam = sam_model_registry[MODEL_TYPE](checkpoint="./model/model_train_2d.pth")
sam.to(device=DEVICE)



mask_predictor = SamPredictor(sam)

image_bgr = cv2.imread("./data/001.png") # bboxなしの画像
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
mask_predictor.set_image(image_rgb)

box = np.array([81, 91, 263, 274]) # これは実際の研究データのbboxを適用している
masks, scores, logits = mask_predictor.predict(
    box=box,
    multimask_output=True
)

print(masks)
print("scores: ",scores)