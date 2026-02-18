from .executor import ExecutorNet
from .modulator import ModulatorNet
from .baselines import build_resnet34

MODEL_REGISTRY = {
    "standalone": lambda config: ExecutorNet().to(config["device"]),
    "resnet34": lambda config: build_resnet34(num_classes=47).to(config["device"]),
    "modulator": lambda config: ModulatorNet(
        ExecutorNet(), rank=config["rank"]
    ).to(config["device"]),
}
