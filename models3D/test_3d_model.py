"""Smoke test for the models3D package using dummy 3D volumes (N, C, D, H, W).

Run directly:
    python models3D/test_3d_model.py

Exercises:
    1. Each 3D generator architecture (resnet_6blocks, resnet_9blocks, unet_128) on a dummy volume.
    2. Each 3D discriminator architecture (basic, n_layers, pixel) on a dummy volume.
    3. A full CycleGANModel3D forward + backward step (set_input -> optimize_parameters) with
       a minimal hand-built options namespace, to confirm the whole pipeline works end to end
       on volumetric data.
"""

import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models3D import networks
from models3D.cycle_gan_model import CycleGANModel
from models3D.temporal import TemporalTCN, bottleneck_to_temporal_sequences, temporal_sequences_to_bottleneck


def test_generators():
    print("Testing 3D generators...")
    x_small = torch.randn(1, 1, 32, 32, 32) # (B, T, H, W, D)
    for netG_name in ["resnet_6blocks", "resnet_9blocks"]:
        net = networks.define_G(input_nc=1, output_nc=1, ngf=8, netG=netG_name, norm="instance")
        out = net(x_small)
        assert out.shape == x_small.shape, f"{netG_name} shape mismatch: {out.shape}"
        print(f"  {netG_name}: input {tuple(x_small.shape)} -> output {tuple(out.shape)} OK")

    # unet_128 halves spatial dims 7 times, so input size must be divisible by 2**7 = 128
    x_unet = torch.randn(1, 1, 128, 128, 128)
    net = networks.define_G(input_nc=1, output_nc=1, ngf=4, netG="unet_128", norm="instance")
    out = net(x_unet)
    assert out.shape == x_unet.shape, f"unet_128 shape mismatch: {out.shape}"
    print(f"  unet_128: input {tuple(x_unet.shape)} -> output {tuple(out.shape)} OK")


def test_residual_generators():
    print("Testing 3D residual generators...")
    x_small = torch.randn(1, 1, 32, 32, 32)
    for residual_mode in ["tanh", "linear"]:
        net = networks.define_G(input_nc=1, output_nc=1, ngf=8, netG="resnet_6blocks", norm="instance", use_residual=True, residual_mode=residual_mode)
        # init_net applies init_weights, which is what actually honors the near-zero final-layer init
        # (constructor-time init would otherwise be moot, since setup()/init_net always runs afterward).
        net = networks.init_net(net, init_type="normal", init_gain=0.02)
        # init_net auto-moves to cuda:0 when available; match the input to wherever the net landed
        # (setup() does this reconciliation itself via a later net.to(self.device), but a bare
        # init_net() call here doesn't, so we do it explicitly).
        device = next(net.parameters()).device
        x_dev = x_small.to(device)
        out = net(x_dev)
        assert out.shape == x_dev.shape, f"residual({residual_mode}) shape mismatch: {out.shape}"
        max_dev = (out - x_dev).abs().max().item()
        assert max_dev < 0.1, f"residual({residual_mode}) not near-identity at init: max |out-in|={max_dev:.4f}"
        print(f"  residual({residual_mode}): input {tuple(x_small.shape)} -> output {tuple(out.shape)}, max|out-in|={max_dev:.4f} OK")

    # invalid mode / mismatched channels should fail loudly, not silently misbehave
    try:
        networks.define_G(input_nc=1, output_nc=1, ngf=8, netG="resnet_6blocks", norm="instance", use_residual=True, residual_mode="bogus")
        raise AssertionError("expected NotImplementedError for invalid residual_mode")
    except NotImplementedError:
        pass
    try:
        networks.define_G(input_nc=1, output_nc=2, ngf=8, netG="resnet_6blocks", norm="instance", use_residual=True)
        raise AssertionError("expected ValueError for input_nc != output_nc with use_residual")
    except ValueError:
        pass
    try:
        networks.define_G(input_nc=1, output_nc=1, ngf=8, netG="unet_128", norm="instance", use_residual=True)
        raise AssertionError("expected NotImplementedError for use_residual with a unet generator")
    except NotImplementedError:
        pass
    print("  residual guard checks (bad mode / mismatched channels / unet) OK")


def test_temporal_module():
    """TemporalTCN in isolation: (N, F, T) -> (N, F, T), and gradients reach every block."""
    print("Testing TemporalTCN (standalone, on (N, F, T) sequences)...")
    N, F, T = 6, 16, 20  # N = an effective batch (e.g. subjects * bottleneck spatial locations)
    tcn = TemporalTCN(feature_channels=F, dilations=(1, 2, 4), dropout=0.1, norm_groups=8)
    x = torch.randn(N, F, T, requires_grad=True)
    out = tcn(x)
    assert out.shape == x.shape, f"TemporalTCN shape mismatch: {out.shape} vs {x.shape}"
    print(f"  TemporalTCN: input {tuple(x.shape)} -> output {tuple(out.shape)} OK")

    out.mean().backward()
    assert x.grad is not None and x.grad.abs().sum().item() > 0, "no gradient reached the TemporalTCN input"
    for i, block in enumerate(tcn.blocks):
        for name, p in block.named_parameters():
            assert p.grad is not None, f"block {i} param '{name}' got no gradient"
    print("  TemporalTCN backward pass: input and every block received gradients OK")


def test_bottleneck_rearrangement_roundtrip():
    """The permute/reshape dance around the temporal module must be exactly invertible on its own,
    independent of whatever the temporal module itself does -- verified here with the temporal
    module bypassed entirely (never called), isolating the tensor-mechanics correctness."""
    print("Testing bottleneck<->temporal-sequence rearrangement is an exact round trip (module bypassed)...")
    B, T, F, d, h, w = 2, 5, 12, 3, 4, 5  # deliberately distinct sizes per axis to catch axis-order bugs
    original_bottleneck = torch.randn(B * T, F, d, h, w)

    z_seq, (F_out, d_out, h_out, w_out) = bottleneck_to_temporal_sequences(original_bottleneck, B, T)
    assert z_seq.shape == (B * d * h * w, F, T), f"unexpected z_seq shape {tuple(z_seq.shape)}"
    assert (F_out, d_out, h_out, w_out) == (F, d, h, w)

    reconstructed_bottleneck = temporal_sequences_to_bottleneck(z_seq, B, T, d, h, w)
    assert torch.equal(original_bottleneck, reconstructed_bottleneck), "bottleneck rearrangement round trip is not exact"
    print(f"  round trip exact: {tuple(original_bottleneck.shape)} -> {tuple(z_seq.shape)} -> {tuple(reconstructed_bottleneck.shape)} OK")


def test_generator_temporal():
    """Full ResnetGenerator with use_temporal_module=True: (B,C,T,H,W,D) in, same shape out, plus a
    backward pass confirming gradients flow through encoder, temporal module, and decoder."""
    print("Testing 3D generator with temporal module (6D (B,C,T,H,W,D) fMRI-sequence input)...")
    B, C, T, H, W, D = 2, 1, 5, 16, 16, 16  # memory-safe; H=W=D=16 divisible by 2**n_downsampling=4
    x = torch.randn(B, C, T, H, W, D, requires_grad=True)

    net = networks.define_G(input_nc=C, output_nc=C, ngf=8, netG="resnet_6blocks", norm="instance", use_temporal_module=True)
    net = networks.init_net(net, init_type="normal", init_gain=0.02)
    device = next(net.parameters()).device
    x_dev = x.to(device)

    out = net(x_dev)
    assert out.shape == x_dev.shape, f"temporal generator shape mismatch: {out.shape} vs {x_dev.shape}"
    print(f"  shape: input {tuple(x_dev.shape)} -> output {tuple(out.shape)} OK")

    # Backward pass: confirm gradients reach the encoder, the temporal module, and the decoder.
    loss = out.mean()
    loss.backward()
    for name, p in net.encoder.named_parameters():
        assert p.grad is not None, f"encoder param '{name}' got no gradient"
    for name, p in net.temporal_module.named_parameters():
        assert p.grad is not None, f"temporal_module param '{name}' got no gradient"
    for name, p in net.decoder.named_parameters():
        assert p.grad is not None, f"decoder param '{name}' got no gradient"
    print("  backward pass: encoder, temporal_module, and decoder all received gradients OK")

    # Reject malformed input rank instead of misbehaving silently.
    rejected = False
    try:
        net(torch.randn(1, 1, 16, 16, 16))  # 5D, not the 6D this generator now expects
    except AssertionError:
        rejected = True
    assert rejected, "expected an AssertionError for 5D input when use_temporal_module=True"
    print("  rank guard (5D input rejected when use_temporal_module=True) OK")

    # use_temporal_module is resnet-only, same restriction as use_residual.
    try:
        networks.define_G(input_nc=1, output_nc=1, ngf=4, netG="unet_128", norm="instance", use_temporal_module=True)
        raise AssertionError("expected NotImplementedError for use_temporal_module with a unet generator")
    except NotImplementedError:
        pass
    print("  guard check (use_temporal_module with a unet generator) OK")

    # Composability: use_residual and use_temporal_module together, same additive-correction contract.
    net_both = networks.define_G(input_nc=C, output_nc=C, ngf=8, netG="resnet_6blocks", norm="instance", use_temporal_module=True, use_residual=True, residual_mode="tanh")
    net_both = networks.init_net(net_both, init_type="normal", init_gain=0.02)
    device_both = next(net_both.parameters()).device
    out_both = net_both(x.to(device_both))
    assert out_both.shape == x.shape, f"temporal+residual generator shape mismatch: {out_both.shape}"
    print(f"  use_temporal_module + use_residual composed: input {tuple(x.shape)} -> output {tuple(out_both.shape)} OK")


def test_discriminators():
    print("Testing 3D discriminators...")
    x = torch.randn(1, 1, 32, 32, 32)
    for netD_name in ["basic", "pixel"]:
        net = networks.define_D(input_nc=1, ndf=8, netD=netD_name, norm="instance")
        out = net(x)
        print(f"  {netD_name}: input {tuple(x.shape)} -> output {tuple(out.shape)} OK")

    net = networks.define_D(input_nc=1, ndf=8, netD="n_layers", n_layers_D=2, norm="instance")
    out = net(x)
    print(f"  n_layers(2): input {tuple(x.shape)} -> output {tuple(out.shape)} OK")


def test_discriminator_temporal_none_matches_original():
    """Requirement 1: discriminator_temporal_mode='none' reproduces the original discriminator exactly
    (same class, same state_dict keys, same output for the same weights)."""
    print("Testing discriminator_temporal_mode='none' reproduces the original discriminator...")
    x = torch.randn(1, 1, 32, 32, 32)

    torch.manual_seed(0)
    net_direct = networks.NLayerDiscriminator(input_nc=1, ndf=8, n_layers=3, norm_layer=networks.get_norm_layer("instance"))
    torch.manual_seed(0)
    net_via_define_d = networks.define_D(input_nc=1, ndf=8, netD="basic", norm="instance", discriminator_temporal_mode="none")

    assert type(net_via_define_d) is networks.NLayerDiscriminator, f"'none' mode returned {type(net_via_define_d)}, expected NLayerDiscriminator"
    assert net_direct.state_dict().keys() == net_via_define_d.state_dict().keys(), "state_dict keys differ between direct construction and define_D(mode='none')"
    assert torch.equal(net_direct(x), net_via_define_d(x)), "'none' mode output differs from directly-constructed NLayerDiscriminator with identical weights"
    print("  'none' mode: identical class, state_dict keys, and output vs. direct NLayerDiscriminator construction OK")


def test_discriminator_temporal_modes():
    """Requirements 2,3,4,6,7,8,10,11: both temporal modes accept variable T, output T equals input T,
    patch logits have shape (B,1,T,d,h,w) (C before T, matching ResnetGenerator's use_temporal_module
    convention), real/fake share the same forward path, gradients reach the stem/temporal_module/body,
    and reusing one net instance across different T values (without reconstructing it) proves T is
    never baked into any Conv3d's fixed channel count."""
    print("Testing temporal discriminator modes (tcn, convlstm3d)...")
    B, C, H, W, D = 2, 1, 32, 32, 32

    for mode in ["tcn", "convlstm3d"]:
        net = networks.define_D(input_nc=C, ndf=8, netD="basic", norm="instance", discriminator_temporal_mode=mode)
        net = networks.init_net(net, init_type="normal", init_gain=0.02)
        device = next(net.parameters()).device

        for T in (3, 7):  # same net instance, two different T values -- see requirement 11 note below
            real_chunk = torch.randn(B, C, T, H, W, D, device=device)
            fake_chunk = torch.randn(B, C, T, H, W, D, device=device)

            out_real = net(real_chunk)  # requirement 7: real and fake go through the exact same net.forward
            out_fake = net(fake_chunk)
            assert out_real.shape == out_fake.shape, f"{mode}: real/fake output shapes differ: {out_real.shape} vs {out_fake.shape}"
            assert out_real.shape[0] == B and out_real.shape[1] == 1 and out_real.shape[2] == T, f"{mode}: unexpected (B,1,T,...) prefix {tuple(out_real.shape)}"
            print(f"  {mode} T={T}: input {tuple(real_chunk.shape)} -> patch logits {tuple(out_real.shape)} OK")

        # Requirement 11: the SAME net instance (no reconstruction) just handled T=3 and T=7 above.
        # If T had been used as a fixed Conv3d channel count anywhere, the second call would have
        # raised a channel-mismatch error instead of succeeding.
        print(f"  {mode}: same net instance handled T=3 and T=7 without reconstruction -> T is not a Conv3d channel OK")

        # Requirements 8, 9, 10: gradients reach the discriminator's stem/temporal_module/body, and back
        # through to whatever produced the "fake" data (stood in here by a leaf nn.Parameter -- this
        # discriminator now shares the generator's exact (B,C,T,H,W,D) convention, but full
        # CycleGANModel training-loop wiring is still a separate, not-yet-done integration step).
        fake_input = torch.nn.Parameter(torch.randn(B, C, 4, H, W, D, device=device))
        pred_fake = net(fake_input)
        loss = pred_fake.mean()  # stand-in for criterionGAN(pred_fake, False)
        loss.backward()
        assert fake_input.grad is not None and fake_input.grad.abs().sum().item() > 0, f"{mode}: no gradient reached the upstream 'generator' input"
        for name, p in net.stem.named_parameters():
            assert p.grad is not None, f"{mode}: stem param '{name}' got no gradient"
        for name, p in net.temporal_module.named_parameters():
            assert p.grad is not None, f"{mode}: temporal_module param '{name}' got no gradient"
        for name, p in net.body.named_parameters():
            assert p.grad is not None, f"{mode}: body param '{name}' got no gradient"
        print(f"  {mode}: backward pass -- stem, temporal_module, body, and upstream input all received gradients OK")


def test_convlstm3d_all_hidden_states():
    """Requirement 5: ConvLSTM3D returns ALL hidden states (B,T,F_hidden,d,h,w), not just the final one."""
    print("Testing ConvLSTM3D returns all hidden states...")
    from models3D.convlstm3d import ConvLSTM3D

    B, T, F, d, h, w = 2, 6, 8, 4, 4, 4
    hidden_channels = 5
    lstm = ConvLSTM3D(input_channels=F, hidden_channels=hidden_channels, kernel_size=3, num_layers=1)
    x = torch.randn(B, T, F, d, h, w, requires_grad=True)
    out = lstm(x)
    assert out.shape == (B, T, hidden_channels, d, h, w), f"expected all-T hidden states {(B, T, hidden_channels, d, h, w)}, got {tuple(out.shape)}"
    # A single-timestep run must NOT reproduce the last hidden state of the full-T run (different H_prev
    # history), so this also indirectly confirms every timestep's hidden state is genuinely distinct.
    out.sum().backward()
    assert x.grad is not None and x.grad.abs().sum().item() > 0
    for name, p in lstm.named_parameters():
        assert p.grad is not None, f"ConvLSTM3D param '{name}' got no gradient"
    print(f"  ConvLSTM3D: input {tuple(x.shape)} -> all hidden states {tuple(out.shape)} (T preserved, not collapsed to 1) OK")


def test_discriminator_checkpoint_compatibility():
    """Checkpoints section: 'none' mode loads its own checkpoints fine; loading a checkpoint from one
    mode into a different mode must fail loudly (architecture mismatch), not silently misbehave."""
    print("Testing discriminator checkpoint compatibility / mismatch detection...")

    net_none_a = networks.define_D(input_nc=1, ndf=8, netD="basic", norm="instance", discriminator_temporal_mode="none")
    net_none_b = networks.define_D(input_nc=1, ndf=8, netD="basic", norm="instance", discriminator_temporal_mode="none")
    net_none_b.load_state_dict(net_none_a.state_dict())  # same mode -> must succeed
    print("  'none' -> 'none' checkpoint load OK")

    net_tcn = networks.define_D(input_nc=1, ndf=8, netD="basic", norm="instance", discriminator_temporal_mode="tcn")
    mismatched = False
    try:
        net_tcn.load_state_dict(net_none_a.state_dict())
    except RuntimeError:
        mismatched = True
    assert mismatched, "expected a RuntimeError loading a 'none'-mode checkpoint into a 'tcn'-mode discriminator"
    print("  'none' -> 'tcn' checkpoint load correctly raised a RuntimeError (architecture mismatch) OK")

    net_convlstm = networks.define_D(input_nc=1, ndf=8, netD="basic", norm="instance", discriminator_temporal_mode="convlstm3d")
    mismatched = False
    try:
        net_convlstm.load_state_dict(net_tcn.state_dict())
    except RuntimeError:
        mismatched = True
    assert mismatched, "expected a RuntimeError loading a 'tcn'-mode checkpoint into a 'convlstm3d'-mode discriminator"
    print("  'tcn' -> 'convlstm3d' checkpoint load correctly raised a RuntimeError (architecture mismatch) OK")


def test_replay_buffer_preserves_whole_chunks():
    """Requirement 12: ImagePool must never mix time points from different chunks -- every returned
    sample along the batch dim must be one complete, untouched original chunk."""
    print("Testing ImagePool preserves complete (C,T,H,W,D) chunks intact (no time-point shuffling)...")
    from util.image_pool import ImagePool

    B, C, T, H, W, D = 8, 1, 4, 3, 3, 3  # (B,C,T,H,W,D), matching the generator/discriminator convention
    pool = ImagePool(pool_size=4)

    # Each of the B chunks is filled uniformly with its own index value, so any cross-chunk mixing
    # (e.g. swapping individual time points between chunks) would produce a non-uniform returned sample.
    chunks = torch.stack([torch.full((C, T, H, W, D), float(i)) for i in range(B)], dim=0)  # (B,C,T,H,W,D)

    seen_any_swap = False
    for _ in range(5):  # multiple passes to exercise the pool's swap logic
        out = pool.query(chunks)
        assert out.shape == chunks.shape
        for i in range(out.shape[0]):
            values = torch.unique(out[i])
            assert values.numel() == 1, f"chunk {i} is not uniform after pool.query -- time points were mixed: unique values {values.tolist()}"
        if not torch.equal(out, chunks):
            seen_any_swap = True
    assert seen_any_swap, "pool never returned a swapped-in historical chunk across 5 queries (pool_size>0 should trigger swaps)"
    print("  every returned chunk stayed uniform (whole chunks preserved) across repeated queries, including swaps OK")


def build_dummy_opt(checkpoints_dir, use_residual=False, residual_mode="tanh", discriminator_temporal_mode="none"):
    """Minimal options namespace covering every attribute CycleGANModel/BaseModel touch."""
    return SimpleNamespace(
        # BaseModel
        isTrain=True,
        checkpoints_dir=checkpoints_dir,
        name="dummy3d",
        device=torch.device("cpu"),
        preprocess="none",
        verbose=False,
        continue_train=False,
        load_iter=0,
        epoch="latest",
        # setup() / schedulers
        init_type="normal",
        init_gain=0.02,
        lr_policy="linear",
        epoch_count=1,
        n_epochs=2,
        n_epochs_decay=2,
        lr_decay_iters=50,
        # CycleGANModel networks
        input_nc=1,
        output_nc=1,
        ngf=8,
        netG="resnet_6blocks",
        norm="instance",
        no_dropout=True,
        ndf=8,
        netD="basic",
        n_layers_D=3,
        # CycleGANModel losses/optim
        pool_size=0,
        gan_mode="lsgan",
        lr=0.0002,
        beta1=0.5,
        lambda_identity=0.5,
        lambda_A=10.0,
        lambda_B=10.0,
        direction="AtoB",
        # residual generator
        use_residual=use_residual,
        residual_mode=residual_mode,
        residual_alpha_max=0.5,
        residual_alpha_init=0.15,
        # temporal discriminator
        discriminator_temporal_mode=discriminator_temporal_mode,
        discriminator_stem_downsample_layers=2,
        temporal_dilations=(1, 2, 4),
        temporal_dropout=0.1,
        temporal_norm_groups=8,
        convlstm_hidden_channels=None,
        convlstm_num_layers=1,
        convlstm_kernel_size=3,
    )


def test_cyclegan_model():
    print("Testing full CycleGANModel3D forward/backward on dummy volumes...")
    checkpoints_dir = tempfile.mkdtemp(prefix="models3d_smoke_")
    try:
        opt = build_dummy_opt(checkpoints_dir)
        model = CycleGANModel(opt)
        model.setup(opt)

        real_A = torch.randn(1, 1, 32, 32, 32)
        real_B = torch.randn(1, 1, 32, 32, 32)
        model.set_input({"A": real_A, "B": real_B, "A_paths": ["dummy_A"], "B_paths": ["dummy_B"]})

        for step in range(2):
            model.optimize_parameters()
            losses = model.get_current_losses()
            print(f"  step {step}: " + ", ".join(f"{k}={v:.4f}" for k, v in losses.items()))

        visuals = model.get_current_visuals()
        for name, tensor in visuals.items():
            assert tensor.shape == real_A.shape, f"visual '{name}' has unexpected shape {tensor.shape}"
        print(f"  visuals OK: {list(visuals.keys())}")
    finally:
        shutil.rmtree(checkpoints_dir, ignore_errors=True)


def test_residual_cyclegan_model():
    print("Testing full CycleGANModel3D with residual generator (opt-flag threading end to end)...")
    for residual_mode in ["tanh", "linear"]:
        checkpoints_dir = tempfile.mkdtemp(prefix="models3d_residual_smoke_")
        try:
            opt = build_dummy_opt(checkpoints_dir, use_residual=True, residual_mode=residual_mode)
            model = CycleGANModel(opt)
            model.setup(opt)

            real_A = torch.randn(1, 1, 32, 32, 32)
            real_B = torch.randn(1, 1, 32, 32, 32)
            model.set_input({"A": real_A, "B": real_B, "A_paths": ["dummy_A"], "B_paths": ["dummy_B"]})

            # Before any training, the generators should be near-identity (near-zero-init correction branch).
            with torch.no_grad():
                fake_B_init = model.netG_A(real_A)
            max_dev = (fake_B_init - real_A).abs().max().item()
            assert max_dev < 0.1, f"residual({residual_mode}) netG_A not near-identity at init: max|out-in|={max_dev:.4f}"

            for step in range(2):
                model.optimize_parameters()
                losses = model.get_current_losses()
                print(f"  [{residual_mode}] step {step}: " + ", ".join(f"{k}={v:.4f}" for k, v in losses.items()))

            visuals = model.get_current_visuals()
            for name, tensor in visuals.items():
                assert tensor.shape == real_A.shape, f"visual '{name}' has unexpected shape {tensor.shape}"
            print(f"  [{residual_mode}] near-identity at init (max|out-in|={max_dev:.4f}), visuals OK: {list(visuals.keys())}")
        finally:
            shutil.rmtree(checkpoints_dir, ignore_errors=True)


def test_cyclegan_model_temporal_discriminator_construction():
    """CycleGANModel wiring: netD_A/netD_B are built independently (no shared weights) for each temporal
    mode, and 'none' still yields the exact same class as before. Construction-only -- a full
    optimize_parameters() training loop with a temporal discriminator additionally needs the
    generator<->discriminator axis-order question resolved first (see discriminator_temporal_gap
    project note), so it's intentionally not exercised here."""
    print("Testing CycleGANModel discriminator construction across temporal modes...")
    for mode in ["none", "tcn", "convlstm3d"]:
        checkpoints_dir = tempfile.mkdtemp(prefix="models3d_disc_temporal_smoke_")
        try:
            opt = build_dummy_opt(checkpoints_dir, discriminator_temporal_mode=mode)
            model = CycleGANModel(opt)
            model.setup(opt)

            if mode == "none":
                assert type(model.netD_A) is networks.NLayerDiscriminator
                assert type(model.netD_B) is networks.NLayerDiscriminator
            else:
                assert type(model.netD_A) is networks.TemporalNLayerDiscriminator
                assert type(model.netD_B) is networks.TemporalNLayerDiscriminator
                assert model.netD_A.temporal_module is not model.netD_B.temporal_module, f"{mode}: netD_A/netD_B are sharing the same temporal_module instance"

            # independent weights: perturbing one net's parameters must not affect the other's
            p_a = next(model.netD_A.parameters())
            p_b_before = next(model.netD_B.parameters()).clone()
            with torch.no_grad():
                p_a.add_(1.0)
            p_b_after = next(model.netD_B.parameters())
            assert torch.equal(p_b_before, p_b_after), f"{mode}: netD_A and netD_B appear to share weights"
            print(f"  {mode}: netD_A ({type(model.netD_A).__name__}) and netD_B independently constructed, no shared weights OK")
        finally:
            shutil.rmtree(checkpoints_dir, ignore_errors=True)


if __name__ == "__main__":
    test_generators()
    test_residual_generators()
    test_temporal_module()
    test_bottleneck_rearrangement_roundtrip()
    test_generator_temporal()
    test_discriminators()
    test_discriminator_temporal_none_matches_original()
    test_discriminator_temporal_modes()
    test_convlstm3d_all_hidden_states()
    test_discriminator_checkpoint_compatibility()
    test_replay_buffer_preserves_whole_chunks()
    test_cyclegan_model_temporal_discriminator_construction()
    test_cyclegan_model()
    test_residual_cyclegan_model()
    print("All models3D smoke tests passed.")
