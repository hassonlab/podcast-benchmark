import torch

from models.shared_decoders import MLPProbeDecoder


def test_mlp_probe_input_dropout_is_applied_only_during_training():
    model = MLPProbeDecoder(
        input_dim=3, layer_sizes=[1], input_dropout=1.0, output_dim=1
    )
    with torch.no_grad():
        model.layers[0].weight.fill_(1.0)
        model.layers[0].bias.zero_()

    inputs = torch.ones(2, 3)
    model.train()
    torch.testing.assert_close(model(inputs), torch.zeros(2))
    model.eval()
    torch.testing.assert_close(model(inputs), torch.full((2,), 3.0))
