import base64
from io import BytesIO

import numpy
import torch
from torch import nn, optim
from torchvision import transforms


class Generator:

    def __init__(self, model, weights_file):
        self.model = model
        checkpoint = torch.load(weights_file)
        self.model.load_state_dict(checkpoint["model_state_dict"])

    def generate(self, z: "numpy.ndarray", base64=False) -> "numpy.ndarray":
        with torch.no_grad():
            samples = self.model.decode(z).cpu()
        
        print(f"images generated! with shape{sample.size()}")

        transformer = transforms.ToPILImage()

        if base64:
            buffered = BytesIO()
            base64str_images = []
            for sample in samples:
                image = transformer(sample).convert("RGB")
                image.save(buffered, format="JPEG")
                base64str_images.append(base64.b64encode(buffered.getvalue()))
            return base64str_images
        return [transformer(t).convert("RGB") for t in samples]


if __name__ == "__main__":
    model = CVAE(100).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
