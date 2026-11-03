import cv2, os, torch, numpy as np
from model import FastSegUNet

model = FastSegUNet(base_ch=24)
state = torch.load('weights_1.pt', map_location='cpu', weights_only=True)
model.load_state_dict({k: v.float() for k, v in state.items()})
model.eval()

for stem in ['photo6', 'photo7', 'photo8']:
    img = cv2.imread(f'captured/images_256/{stem}.png')
    gt = (cv2.imread(f'captured/masks_256/{stem}.png', cv2.IMREAD_GRAYSCALE) >= 128).astype(np.uint8)
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    with torch.no_grad():
        pred = (model(x)[0, 0] > 0).to(torch.uint8).numpy()
    
    gt_obj = gt.sum()
    pred_obj = pred.sum()
    tp = (pred & gt).sum()
    fp = (pred & (~gt)).sum()
    fn = ((~pred) & gt).sum()
    iou_obj = tp / max(1, tp + fp + fn)
    print(f"{stem}: GT obj={gt_obj}, Pred obj={pred_obj}, TP={tp}, FP={fp}, FN={fn}, IoU_obj={iou_obj:.4f}")

