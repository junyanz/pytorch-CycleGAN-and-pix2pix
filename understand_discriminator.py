"""Walk a dummy tensor through NLayerDiscriminator (models/networks.py) and print
the shape / param count after every layer, so you can see how it shrinks the
image down to a grid of real/fake patch predictions instead of a single scalar.

Run:
    python understand_discriminator.py
"""
import torch

from models.networks import NLayerDiscriminator, get_norm_layer

# ---------------------------------------------------------------------------
# Config -- mirrors the CycleGAN defaults in options/base_options.py and
# models/cycle_gan_model.py. Change these to match your actual training
# setup (e.g. input_nc=1 for grayscale MRI/motion-correction images).
# ---------------------------------------------------------------------------
INPUT_NC = 3        # --input_nc  (channels in)
NDF = 64            # --ndf       (filters in the first conv layer)
N_LAYERS = 3        # --n_layers_D (PatchGAN default)
NORM = "instance"   # --norm      (CycleGAN default)
IMG_SIZE = 256      # --crop_size default
BATCH_SIZE = 1


def describe_layer(layer: torch.nn.Module) -> str:
    """Return a short human label for what this layer is doing structurally."""
    if isinstance(layer, torch.nn.Conv2d) and layer.stride == (2, 2):
        return "↓ downsample (stride-2 conv)"
    if isinstance(layer, torch.nn.Conv2d) and layer.out_channels == 1:
        return "  final 1-channel conv -> real/fake logit per patch"
    if isinstance(layer, torch.nn.Conv2d):
        return "  conv (channel projection, stride-1)"
    if isinstance(layer, torch.nn.LeakyReLU):
        return "  activation"
    return ""


def main():
    norm_layer = get_norm_layer(norm_type=NORM)
    net = NLayerDiscriminator(
        input_nc=INPUT_NC,
        ndf=NDF,
        n_layers=N_LAYERS,
        norm_layer=norm_layer,
    )
    net.eval()  # no need to track grads / use batch stats for this inspection

    total_params = sum(p.numel() for p in net.parameters())
    print(f"NLayerDiscriminator: input_nc={INPUT_NC}, ndf={NDF}, "
          f"n_layers={N_LAYERS}, norm={NORM}")
    print(f"Total parameters: {total_params:,}\n")

    dummy = torch.randn(BATCH_SIZE, INPUT_NC, IMG_SIZE, IMG_SIZE)
    print(f"Input shape:  {tuple(dummy.shape)}  (batch, channels, height, width)\n")

    # net.model is a single nn.Sequential holding every layer in order
    # (see NLayerDiscriminator.__init__ in models/networks.py). Register a
    # forward hook on each top-level child to snapshot its output shape.
    header = f"{'#':>3}  {'layer':<45} {'output shape':<22} {'params':>10}  note"
    print(header)
    print("-" * len(header))

    def make_hook(idx, layer):
        def hook(module, inp, out):
            n_params = sum(p.numel() for p in module.parameters())
            note = describe_layer(layer)
            print(f"{idx:>3}  {layer.__class__.__name__:<45} {str(tuple(out.shape)):<22} {n_params:>10,}  {note}")
        return hook

    handles = []
    for idx, layer in enumerate(net.model):
        handles.append(layer.register_forward_hook(make_hook(idx, layer)))

    with torch.no_grad():
        output = net(dummy)

    for h in handles:
        h.remove()

    print("-" * len(header))
    print(f"\nOutput shape: {tuple(output.shape)}  (batch, 1, patch_h, patch_w)")
    print("Each element in the output grid is a real/fake logit for one overlapping "
          "receptive-field patch of the input -- this is the 'PatchGAN' idea: "
          "instead of one real/fake score for the whole image, the discriminator "
          "scores many local patches, which pushes the generator to get local "
          "texture/style right everywhere rather than just gross global structure.")


if __name__ == "__main__":
    main()
