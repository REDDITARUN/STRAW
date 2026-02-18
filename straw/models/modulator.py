"""ModulatorNet -- hypernetwork that generates per-sample low-rank weights
for an ExecutorNet, conditioned on a style encoding of the input image."""

import torch
import torch.nn as nn
from torch.func import functional_call, vmap


class ModulatorNet(nn.Module):
    """Hypernetwork that observes each input image via a style encoder,
    then generates low-rank weight deltas for an executor network.

    The generated parameters are reconstructed from compact (A, B) factor
    pairs and applied per-sample using ``vmap`` + ``functional_call``.
    """

    def __init__(self, executor_model: nn.Module, rank: int = 16):
        super().__init__()
        self.executor_model = executor_model
        self.rank = rank

        # Freeze executor -- it's only an architecture template.
        # functional_call overrides its weights with generated ones,
        # so the executor's own parameters should never be updated.
        for param in self.executor_model.parameters():
            param.requires_grad = False

        # Style encoder: input image -> latent style vector
        self.style_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.Tanh(),
        )

        # Compute the low-rank parameter budget for every executor layer
        self.target_shapes = {
            k: v.shape for k, v in executor_model.named_parameters()
        }
        self.layer_configs = []
        total_params = 0

        for name, shape in self.target_shapes.items():
            if len(shape) > 1:  # weight matrix / tensor
                out_d = shape[0]
                in_d = shape[1:].numel()
                count = (out_d * rank) + (rank * in_d)
                self.layer_configs.append(
                    {
                        "name": name,
                        "type": "w",
                        "shape": shape,
                        "dims": (out_d, in_d),
                        "count": count,
                    }
                )
            else:  # bias vector
                count = shape.numel()
                self.layer_configs.append(
                    {"name": name, "type": "b", "shape": shape, "count": count}
                )
            total_params += count

        # Generator: style vector -> flat compressed parameter vector
        self.generator = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, total_params),
        )

    # ------------------------------------------------------------------
    # Low-rank reconstruction
    # ------------------------------------------------------------------

    def get_weights_for_single_sample(self, style_vector: torch.Tensor):
        """Decode a single style vector into a full set of executor weights."""
        flat_params = self.generator(style_vector)
        weights = {}
        curr = 0

        for config in self.layer_configs:
            count = config["count"]
            chunk = flat_params[curr : curr + count]
            curr += count

            if config["type"] == "w":
                out_d, in_d = config["dims"]
                size_a = out_d * self.rank

                mat_a = chunk[:size_a].view(out_d, self.rank)
                mat_b = chunk[size_a:].view(self.rank, in_d)

                w = torch.matmul(mat_a, mat_b)
                weights[config["name"]] = w.view(config["shape"])
            else:
                weights[config["name"]] = chunk.view(config["shape"])

        return weights

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        style_vectors = self.style_encoder(images)

        def executor_fwd_pass(params, single_input):
            return functional_call(
                self.executor_model, params, single_input.unsqueeze(0)
            ).squeeze(0)

        batch_weights = vmap(self.get_weights_for_single_sample)(style_vectors)
        outputs = vmap(executor_fwd_pass, randomness="different")(
            batch_weights, images
        )
        return outputs
