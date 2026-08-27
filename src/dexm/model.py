import torch
import torch.nn as nn
from torchvision.transforms import v2
from torch.distributions.normal import Normal

from pathlib import Path

DINO_REPO = Path(__file__).resolve().parents[2] / "dinov3"


class Actor(torch.nn.Module):
    def __init__(
        self,
        action_dim: int = 18,
        weights_path: str = "/home/run/Downloads/dinov3_vit7b16.pth",
        resize_size: int = 224,
        dtype: torch.dtype = torch.bfloat16,
        obs_dim: int = 18,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
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

        self.vision_proj = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.LayerNorm(1024),
            nn.ELU(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ELU(),
        ).to(self.device)

        self.actor = nn.Sequential(
            nn.Linear(256 + obs_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, action_dim),
        ).to(self.device)

        self.actor_logstd = nn.Parameter(torch.zeros(action_dim, device=self.device))

    def forward(self, x: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:

        x = self._transform_input_for_dinov3(x)
        with (
            torch.no_grad(),
            torch.autocast(device_type=self.device.type, dtype=self.dtype),
        ):
            x = self.dinov3_backbone(x).float()
        x = self.vision_proj(x)
        x = torch.cat([x, obs], dim=-1)
        x = self.actor(x)
        return x

    def get_action(self, x: torch.Tensor, obs: torch.Tensor, action: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = self.forward(x, obs)
        std = self.actor_logstd.exp().expand_as(mean)
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return action, log_prob, entropy


class Critic(nn.Module):
    def __init__(
        self,
        critic_dim: int = 151,  # 54 (Robots) + 13 (Box) + 48 (Cable) + 24 (Offsets) + 8 (Forces) + 4 (Phase)
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 3-Layer Value MLP with LayerNorm for stable baseline estimation
        self.critic_net = nn.Sequential(
            nn.Linear(critic_dim, 512),
            nn.LayerNorm(512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1),
        ).to(self.device)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        state: Privileged state vector from Newton physics buffer,
               shape (num_envs, 151), dtype float32.
        Returns:
               Value estimate V(s), shape (num_envs, 1).
        """
        return self.critic_net(state.float().to(self.device))


if __name__ == "__main__":
    model = Actor(
        action_dim=18,
        weights_path="/home/run/Downloads/dinov3_vit7b16.pth",
        obs_dim=18,
    )
    import torchvision.io as io

    img_tensor = io.read_image(
        "/home/run/Project/dexm/_static/renders/frame_settled.jpg"
    )
    if img_tensor.shape[0] == 4:
        img_tensor = img_tensor[:3, :, :]
    input_tensor = img_tensor.unsqueeze(0).to(model.device)
    print("Input tensor shape:", input_tensor.shape)

    with torch.inference_mode():
        output = model(input_tensor, torch.randn(1, 18).to(model.device))
    print("Output tensor shape:", output.shape)
