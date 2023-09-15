from tqdm import tqdm
import network
import utils
import os
import time

from datasets import Cityscapes
from torchvision import transforms

import torch
import torch.nn as nn

from PIL import Image
import matplotlib.pyplot as plt
from glob import glob


def time_synchronized():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def main(inputs, result, model_name, ckpt):
    os.makedirs(result, exist_ok=True)
    decode_fn = Cityscapes.decode_target

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: %s" % device)

    # Setup dataloader
    image_files = []
    if os.path.isdir(inputs):
        for ext in ["png", "jpeg", "jpg", "JPEG"]:
            files = glob(os.path.join(inputs, "**/*.%s" % (ext)), recursive=True)
            if len(files) > 0:
                image_files.extend(files)
    elif os.path.isfile(inputs):
        image_files.append(inputs)

    # Set up model (all models are 'constructed at network.modeling) / output_stride : 8 or 16
    model = network.modeling.__dict__[model_name](num_classes=19, output_stride=16)

    utils.set_bn_momentum(model.backbone, momentum=0.01)

    if ckpt is not None and os.path.isfile(ckpt):
        checkpoint = torch.load(ckpt, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        print("Resume model from %s" % ckpt)
        del checkpoint

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    with torch.no_grad():
        model = model.eval()
        for img_path in tqdm(image_files):
            ext = os.path.basename(img_path).split(".")[-1]
            img_name = os.path.basename(img_path)[: -len(ext) - 1]
            origin_img = Image.open(img_path).convert("RGB")
            img = transform(origin_img).unsqueeze(0)  # To tensor of NCHW
            img = img.to(device)

            t1 = time_synchronized()
            pred = model(img).max(1)[1].cpu().numpy()[0]  # HW
            t2 = time_synchronized()

            print(f"Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference.")

            colorized_preds = decode_fn(pred).astype("uint8")
            colorized_preds = Image.fromarray(colorized_preds)
            colorized_preds = Image.blend(colorized_preds, origin_img, alpha=0.4)

            if result:
                colorized_preds.save(os.path.join(result, img_name + ".png"))

            plt.axis("off")
            plt.imshow(colorized_preds)
            plt.show()


if __name__ == "__main__":
    inputs = "test_inputs/test1.png"
    result = "test_results"

    model_name = "deeplabv3plus_mobilenet"
    ckpt = "weights/best_deeplabv3plus_mobilenet_cityscapes_os16.pth"

    # model_name = "deeplabv3plus_resnet101"
    # ckpt = "weights/best_deeplabv3plus_resnet101_cityscapes_os16.pth"

    main(inputs, result, model_name, ckpt)
