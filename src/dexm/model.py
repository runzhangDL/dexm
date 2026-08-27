import torch
from torchvision.transforms import v2


from pathlib import Path

DINO_REPO = Path(__file__).resolve().parents[2] / "dinov3"


class Action(torch.nn.Module):
    def __init__(
        self, action_dim: int, weights_path: str = "/home/run/Downloads/dinov3_vit7b16.pth", resize_size: int = 224, dtype: torch.dtype = torch.bfloat16
    ):
        super().__init__()
        self.action_dim = action_dim

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        self._transform_input_for_dinov3 = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize((resize_size, resize_size), antialias=True),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
                v2.ToDtype(dtype, scale=False),
            ]
        )

        self.dinov3_backbone = torch.hub.load(
            repo_or_dir=DINO_REPO,
            model="dinov3_vit7b16",
            source="local",
            weights=weights_path,
        ).to(self.device, dtype=dtype)
        self.dinov3_backbone.eval()
        self.dinov3_backbone.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self._transform_input_for_dinov3(x)
        with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=self.dtype):
            x = self.dinov3_backbone(x)
        return x

if __name__ == "__main__":
    model = Action(action_dim=9, weights_path="/home/run/Downloads/dinov3_vit7b16.pth")
    import torchvision.io as io
    img_tensor = io.read_image("/home/run/Project/dexm/_static/renders/frame_settled.jpg")
    if img_tensor.shape[0] == 4:
        img_tensor = img_tensor[:3, :, :]
    input_tensor = img_tensor.unsqueeze(0).to(model.device, dtype=model.dtype)
    print("Input tensor shape:", input_tensor.shape)

    with torch.inference_mode():
        output = model(input_tensor)
    print("Output tensor shape:", output.shape)