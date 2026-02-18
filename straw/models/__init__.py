from .executor import ExecutorNet
from .modulator import ModulatorNet
from .modulator_trm import ModulatorNetTRM
from .baselines import build_resnet34

MODEL_REGISTRY = {
    "standalone": lambda config: ExecutorNet().to(config["device"]),
    "resnet34": lambda config: build_resnet34(num_classes=47).to(config["device"]),
    "modulator": lambda config: ModulatorNet(
        ExecutorNet(), rank=config["rank"]
    ).to(config["device"]),
    "modulator_trm": lambda config: ModulatorNetTRM(
        ExecutorNet(),
        rank=config["rank"],
        trm_token_dim=config.get("trm_token_dim", 64),
        trm_heads=config.get("trm_heads", 4),
        trm_steps=config.get("trm_steps", 4),
        trm_mlp_ratio=config.get("trm_mlp_ratio", 4.0),
        trm_rope_theta=config.get("trm_rope_theta", 10000.0),
        trm_dropout=config.get("trm_dropout", 0.0),
    ).to(config["device"]),
}
