import os, cv2, torch, numpy as np
from model import FastSegUNet
from metrics import batch_bf1, ConfusionAccumulator

model = FastSegUNet(base_ch=24)
state = torch.load('weights_1.pt', map_location='cpu', weights_only=True)
model.load_state_dict({k: v.float() for k, v in state.items()})
model.eval()

img_dir = 'captured/images_256'
mask_dir = 'captured/masks_256'
files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.png','.jpg'))])

total_conf = ConfusionAccumulator()
all_bf1 = []

print("=== Captured Individual Image Evaluation ===")
for f in files:
    stem = os.path.splitext(f)[0]
    img = cv2.imread(os.path.join(img_dir, f))
    gt = cv2.imread(os.path.join(mask_dir, stem + '.png'), cv2.IMREAD_GRAYSCALE)
    gt = (gt >= 128).astype(np.uint8)
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    with torch.no_grad():
        pred = (model(x)[0, 0] > 0).to(torch.uint8).numpy()
    
    conf = ConfusionAccumulator()
    conf.update(pred, gt)
    total_conf.update(pred, gt)
    
    miou = conf.miou()
    bf1 = batch_bf1(pred[None, ...].astype(np.float32), gt[None, ...].astype(np.float32))[0]
    all_bf1.append(bf1)
    q = 0.7 * miou + 0.3 * bf1
    print(f"{stem:8s}: mIoU = {miou:.4f} | BF1 = {bf1:.4f} | Q = {q:.4f}")

overall_miou = total_conf.miou()
overall_bf1 = float(np.mean(all_bf1))
overall_q = 0.7 * overall_miou + 0.3 * overall_bf1
print("-" * 45)
print(f"Overall Captured: mIoU = {overall_miou:.4f} | BF1 = {overall_bf1:.4f} | Q = {overall_q:.4f}")

