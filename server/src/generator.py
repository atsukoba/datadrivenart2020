import base64
from io import BytesIO

import numpy
import torch
from torch import nn, optim
from torchvision import transforms


class Generator:

    def __init__(self, model, weights_file):
        self.model = model
        self.model.load_state_dict(torch.load(weights_file))

    def generate(self, z: list, use_base64=False) -> "numpy.ndarray":
        with torch.no_grad():
            samples = self.model.decode(
                torch.Tensor(z).cpu())

        print(f"images generated! with shape{samples.size()}")

        transformer = transforms.ToPILImage()

        if use_base64:
            base64str_images = []
            for sample in samples:
                buffered = BytesIO()
                image = transformer(sample).convert("RGB")
                print(type(image))
                image.save(buffered, format="JPEG")
                base64str_images.append(base64.b64encode(buffered.getvalue()).decode('utf-8'))
            return base64str_images
        return [transformer(t).convert("RGB") for t in samples]


if __name__ == "__main__":
    model = CVAE(100).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
