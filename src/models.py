"""The two models we compare, built with timm.

Both are pre-trained on ImageNet-21k only (no ImageNet-1k fine-tuning) and
get a new 7-class head:
    vit: ViT-B/16
    bit: BiT ResNet-50 (ResNetV2-50x1)

Usage:
    python src/models.py          # check the names exist and print model sizes
"""

import timm

NUM_CLASSES = 7

# Short key (used in configs) -> exact timm name.
MODEL_NAMES = {
    "vit": "vit_base_patch16_224.augreg_in21k",  # ViT-B/16
    "bit": "resnetv2_50x1_bit.goog_in21k",       # BiT ResNet-50 (ResNetV2-50x1)
}


def check_model_names():
    """Stop with an error if timm does not know one of our model names."""
    for key, name in MODEL_NAMES.items():
        # list_models(pattern, pretrained=True) returns the matching names
        # that have pretrained weights; an empty list means "not found".
        found = timm.list_models(name, pretrained=True)
        assert name in found, f"timm has no pretrained model called {name!r} (key {key!r})"


def create_model(key, pretrained=True):
    """Build one of our models with a fresh 7-class head.

    pretrained=False gives random weights (used by --smoke, so no download).
    """
    check_model_names()
    # num_classes=7 throws away the 21k-class ImageNet head and puts a new,
    # randomly initialised 7-class layer in its place.
    return timm.create_model(MODEL_NAMES[key], pretrained=pretrained, num_classes=NUM_CLASSES)


if __name__ == "__main__":
    check_model_names()
    print("Both model names exist in timm.")
    for key in MODEL_NAMES:
        model = create_model(key, pretrained=False)  # no download needed to count parameters
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{key}: {MODEL_NAMES[key]}  {n_params / 1e6:.1f}M parameters")
