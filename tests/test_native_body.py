import numpy as np
import torch


def test_region_body_tiles_keep_one_native_grain_and_cover_the_full_roi():
    from prototypes.v30_video_apex.native_body import predict_owned_body_region
    class QueryModel(torch.nn.Module):
        image_gain = 1.
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.tensor(0.))
        def forward(self, x):
            # A known function of the native grain-relative coordinates,
            # including parts farther than one 288 px crop can reach.
            p = .1 + .8 * torch.sigmoid(2*x[:, 2:3] - x[:, 3:4] + self.anchor)
            return torch.logit(p)
    gray = np.zeros((560, 640), np.uint8)
    owner = {'grain_native': [305., 281.], 'grain_radius_px': 13}
    roi = [29, 31, 588, 518]
    body, origin, report = predict_owned_body_region(QueryModel(), gray, owner, roi)
    y, x = np.mgrid[31:518, 29:588]
    expected = .1 + .8/(1+np.exp(-(2*np.clip((x-305)/128, -2, 2)-np.clip((y-281)/128, -2, 2))))
    np.testing.assert_allclose(body, expected, atol=2e-7)
    assert origin == roi[:2] and body.shape == (487, 559)
    assert report['covered_pixels'] == body.size and report['overlap_pixels'] > 0
    assert report['max_overlap_probability_disagreement'] < 2e-7
    assert report['same_grain_query'] == owner['grain_native']

from prototypes.v30_video_apex.native_body import (
    NativeBodyNet, body_input, body_loss, load_body_checkpoint, save_body_checkpoint,
    predict_owned_body)


def test_unknown_pixels_have_zero_gradient_with_real_loss():
    logits = torch.zeros((1,1,32,32),requires_grad=True)
    p,n,f = [torch.zeros_like(logits,dtype=torch.bool) for _ in range(3)]
    p[:,:,5:8,5:8]=True
    n[:,:,12:18,12:18]=True
    f[:,:,20:24,20:24]=True
    loss,terms=body_loss(logits,p,n,f)
    loss.backward()
    assert torch.count_nonzero(logits.grad[~(p|n|f)])==0
    assert (logits.grad[p]<0).all() and (logits.grad[n|f]>0).all()
    assert terms["foreign_mass"]==16


def test_same_image_different_grain_queries_and_checkpoint_parity(tmp_path):
    torch.manual_seed(53)
    model=NativeBodyNet(base=4).eval()
    image=np.full((64,64),128,dtype=np.uint8)
    left=body_input(image,[0,0],[16,32])
    right=body_input(image,[0,0],[48,32])
    assert np.array_equal(left[0],right[0])
    with torch.no_grad():
        a=model(torch.from_numpy(left)[None])
        b=model(torch.from_numpy(right)[None])
    assert not torch.allclose(a,b)
    save_body_checkpoint(tmp_path/"body.pt",model,{"test":"same-query-roundtrip"})
    loaded,_=load_body_checkpoint(tmp_path/"body.pt")
    with torch.no_grad():
        torch.testing.assert_close(a,loaded(torch.from_numpy(left)[None]),rtol=0,atol=0)
def test_hard_background_mining_keeps_unknown_gradient_zero():
    import torch
    from prototypes.v30_video_apex.native_body import body_loss
    logits = torch.tensor([[[[1., 4., -6., 20.]]]], requires_grad=True)
    positive = torch.tensor([[[[True, False, False, False]]]])
    ordinary = torch.tensor([[[[False, True, True, False]]]])
    foreign = torch.zeros_like(ordinary)
    loss, terms = body_loss(logits, positive, ordinary, foreign, hard_negative_weight=1)
    loss.backward()
    assert logits.grad[0, 0, 0, 3] == 0
    assert logits.grad[0, 0, 0, 1] > logits.grad[0, 0, 0, 2] > 0
    assert terms["hard_negative_pixels"] == 2


def test_focused_background_budget_strengthens_licensed_errors_without_unknown_loss():
    # Many easy background pixels must not dilute the few wrong positive pixels.
    x = torch.full((1,1,1,1200), -8.)
    x[..., :100] = 4.
    x[..., 100:120] = 3.
    x[..., 1199] = 20.  # highest response is unknown, never mine it
    positive = torch.zeros_like(x, dtype=torch.bool); positive[..., :100] = True
    ordinary = torch.zeros_like(x, dtype=torch.bool); ordinary[..., 100:1100] = True
    foreign = torch.zeros_like(x, dtype=torch.bool)
    gradients, budgets = [], []
    for ratio in (3., .5):
        logits = x.clone().requires_grad_()
        loss, terms = body_loss(logits, positive, ordinary, foreign,
                               hard_negative_weight=.5, hard_negative_ratio=ratio)
        loss.backward()
        gradients.append(logits.grad)
        budgets.append(terms['hard_negative_pixels'])
    assert budgets == [300,64]
    assert gradients[1][..., 100].item() > gradients[0][..., 100].item()
    assert all(torch.count_nonzero(g[~(positive | ordinary | foreign)]) == 0 for g in gradients)
    torch.testing.assert_close(gradients[0][positive], gradients[1][positive])


def test_runtime_uses_the_same_native_crop_and_saved_contrast_gain(tmp_path):
    from prototypes.v30_video_apex.native_caps import extract_tile
    torch.manual_seed(53)
    model = NativeBodyNet(base=4, image_gain=8).eval()
    save_body_checkpoint(tmp_path / "body.pt", model, {"test": "runtime parity"})
    loaded, meta = load_body_checkpoint(tmp_path / "body.pt")
    image = np.random.default_rng(53).integers(90, 150, (350, 350), dtype=np.uint8)
    owner = {"grain_native": [120.25, 155.75], "grain_radius_px": 14}
    for origin in ([17, 23], [-17, -11]):
        direct = body_input(extract_tile([image], origin, 288)[0], origin,
                            owner["grain_native"], 14, image_gain=8)
        with torch.no_grad():
            expected = torch.sigmoid(loaded(torch.from_numpy(direct)[None]))[0, 0].numpy()
        runtime, actual_origin = predict_owned_body(loaded, image, owner, origin)
        assert actual_origin == origin
        np.testing.assert_array_equal(expected, runtime)
    assert meta["config"]["image_contrast_gain_about_midgray"] == 8
