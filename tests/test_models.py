import torch

from vrep.models import ObjectProbe, Predictor, PredictorConfig
from vrep.models.common import FourierFeatures
from vrep.repr.base import TokenBatch


def test_fourier_features():
    ff = FourierFeatures(3, n_freq=4)
    x = torch.rand(2, 7, 3)
    y = ff(x)
    assert y.shape == (2, 7, ff.out_dim) and ff.out_dim == 3 * 9


def test_predictor_2d_residual_zero_init():
    cfg = PredictorConfig(feat_dim=16, pos_dim=2, out_dim=16, d_model=32, n_layers=1, n_heads=2)
    m = Predictor(cfg)
    tb = TokenBatch(torch.randn(3, 10, 16), torch.rand(3, 10, 2))
    out = m(tb, torch.zeros(3, 2))
    assert out.out.shape == (3, 10, 16)
    assert torch.allclose(out.out, torch.zeros_like(out.out))  # zero-init head
    assert out.gate is None and out.memory is None


def test_predictor_respects_mask_and_is_permutation_equivariant():
    cfg = PredictorConfig(feat_dim=5, pos_dim=3, out_dim=3, d_model=32, n_layers=2, n_heads=2, pos_range=0.3)
    m = Predictor(cfg)
    with torch.no_grad():
        m.head.weight.normal_(0, 0.1)
    torch.manual_seed(0)
    feat, pos = torch.randn(1, 8, 5), torch.randn(1, 8, 3) * 0.2
    mask = torch.ones(1, 8, dtype=torch.bool)
    mask[0, 6:] = False
    a = torch.tensor([[0.03, -0.01]])
    out = m(TokenBatch(feat, pos, mask), a).out
    assert out.shape == (1, 8, 3)
    assert torch.all(out[0, 6:] == 0)  # padded tokens produce no output
    # permuting the valid tokens permutes the outputs
    perm = torch.tensor([3, 0, 5, 1, 4, 2, 6, 7])
    out_p = m(TokenBatch(feat[:, perm], pos[:, perm], mask[:, perm]), a).out
    assert torch.allclose(out_p, out[:, perm], atol=1e-5)
    # padded tokens must not influence valid ones
    feat2 = feat.clone()
    feat2[0, 6:] = 100.0
    out2 = m(TokenBatch(feat2, pos, mask), a).out
    assert torch.allclose(out2[:, :6], out[:, :6], atol=1e-5)


def test_predictor_memory_and_gate():
    cfg = PredictorConfig(feat_dim=8, pos_dim=2, out_dim=8, d_model=32, n_layers=1, n_heads=2, n_memory=3, gated=True)
    m = Predictor(cfg)
    tb = TokenBatch(torch.randn(2, 6, 8), torch.rand(2, 6, 2))
    out = m(tb, torch.zeros(2, 2))
    assert out.memory.shape == (2, 3, 32) and out.gate.shape == (2, 6)
    assert torch.all((out.gate > 0) & (out.gate < 1))
    out2 = m(tb, torch.zeros(2, 2), memory=out.memory)
    assert out2.memory.shape == (2, 3, 32)
    assert not torch.allclose(out2.memory, out.memory)


def test_predictor_action_changes_output():
    cfg = PredictorConfig(feat_dim=8, pos_dim=2, out_dim=8, d_model=32, n_layers=1, n_heads=2)
    m = Predictor(cfg)
    with torch.no_grad():
        m.head.weight.normal_(0, 0.1)
    tb = TokenBatch(torch.randn(1, 6, 8), torch.rand(1, 6, 2))
    o1 = m(tb, torch.tensor([[0.05, 0.0]])).out
    o2 = m(tb, torch.tensor([[-0.05, 0.0]])).out
    assert not torch.allclose(o1, o2)


def test_probe_shapes_and_training_step():
    probe = ObjectProbe(feat_dim=8, pos_dim=2, n_objects=3, d_model=32, n_layers=1, n_heads=2)
    tb = TokenBatch(torch.randn(4, 12, 8), torch.rand(4, 12, 2))
    pos, vis = probe(tb)
    assert pos.shape == (4, 3, 2) and vis.shape == (4, 3)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    target = torch.rand(4, 3, 2) * 0.2
    l0 = None
    for _ in range(30):
        pos, _ = probe(tb)
        loss = ((pos - target) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        l0 = l0 or loss.item()
    assert loss.item() < l0


def test_token_decoder_shapes():
    from vrep.models import TokenDecoder

    dec = TokenDecoder(feat_dim=16, grid=4, out_size=32, width=32)
    tb = TokenBatch(torch.randn(2, 16, 16), torch.rand(2, 16, 2))
    img = dec(tb)
    assert img.shape == (2, 32, 32, 3) and torch.isfinite(img).all()
