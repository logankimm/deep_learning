from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        self.model = nn.Sequential(
            nn.Linear(2 * n_track * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_waypoints * 2),
        )


    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        b = track_left.shape[0]
        x = torch.cat([track_left, track_right], dim=1)
        x = x.flatten(start_dim=1)
        x = self.model(x)
        return x.view(b, self.n_waypoints, 2)

        raise NotImplementedError


class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
        n_head: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        self.query_embed = nn.Embedding(n_waypoints, d_model)

        self.input_proj = nn.Linear(2, d_model)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_head, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # 4. An output head to project from d_model to 2D waypoints
        self.output_head = nn.Linear(d_model, 2)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        b = track_left.shape[0]

        # (n_waypoints, d_model) -> (1, n_waypoints, d_model) -> (b, n_waypoints, d_model)
        # create embedding for possible waypoint positions
        queries = self.query_embed.weight.unsqueeze(0).repeat(b, 1, 1)
        # print(queries.shape)

        track_points = torch.cat([track_left, track_right], dim=1)  # (b, 2 * n_track, 2)
        # print(track_points.shape)
        memory = self.input_proj(track_points)  # (b, 2 * n_track, d_model)
        # print(memory.shape)

        output = self.decoder(tgt=queries, memory=memory) # (b, n_waypoints, d_model)
        # print(output.shape)

        # 4. Get final waypoints
        waypoints = self.output_head(output) # (b, n_waypoints, 2)

        return waypoints


class CNNPlanner(torch.nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
    ):
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

        num_channels = 32

        self.blocks = nn.Sequential(
            nn.Conv2d(3, num_channels, kernel_size = 3, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels * 2, kernel_size=3, stride = 2, padding = 1, bias=False), # 32x64
            nn.BatchNorm2d(num_channels * 2),
            nn.ReLU(),
            nn.Conv2d(num_channels * 2, num_channels * 4, kernel_size=3, stride = 2, padding = 1, bias=False), # 64x128
            nn.BatchNorm2d(num_channels * 4),
            nn.ReLU(),
            nn.Conv2d(num_channels * 4, num_channels * 8, kernel_size=3, stride = 2, padding = 1, bias=False), # 64x128
            nn.BatchNorm2d(num_channels * 8),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.outputs = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_waypoints * 2)
        )

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        b = image.shape[0]
        x = (image - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        features = self.blocks(x)
        # print(features.shape)

        output = self.outputs(features)
        
        return output.view(b, self.n_waypoints, 2)

        raise NotImplementedError


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
