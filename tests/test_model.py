import torch.nn as nn

from paww.nn import Model


def test_model():

    cfg = dict(model_name="tf_efficientnet_b0_ns", target_size=1)

    net = Model(cfg, pretrained=False)

    assert isinstance(net, nn.Module)
