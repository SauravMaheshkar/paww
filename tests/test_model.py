import torch.nn as nn

from paww.nn import Model


def test_efficient():

    cfg = dict(
        model_name="tf_efficientnet_b0_ns", model_type="EfficientNet", target_size=1
    )

    net = Model(cfg, pretrained=False)

    assert isinstance(net, nn.Module)


def test_resnet():

    cfg = dict(model_name="resnet18", model_type="ResNet", target_size=1)

    net = Model(cfg, pretrained=False)

    assert isinstance(net, nn.Module)
