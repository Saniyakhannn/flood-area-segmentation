import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from attention_unet_model import AttentionUNet
from unet_flood_model import UNet


class GradCAM:
    def __init__(self, model, target_layer, device):
        self.model = model
        self.device = device
        self.gradients = None
        self.activations = None
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, img_tensor, target_size=(256, 256)):
        self.model.eval()
        img_tensor = img_tensor.to(self.device)
        img_tensor.requires_grad_(True)
        output = self.model(img_tensor)
        score = torch.sigmoid(output).mean()
        self.model.zero_grad()
        score.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=target_size, mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


class AttentionMapExtractor:
    def __init__(self, model):
        self.maps = {}
        self._hooks = []
        for name, module in model.named_modules():
            if "psi" in name and name.endswith("2"):
                gate_name = name.split(".")[0]
                hook = module.register_forward_hook(self._make_hook(gate_name))
                self._hooks.append(hook)

    def _make_hook(self, gate_name):
        def hook(module, input, output):
            self.maps[gate_name] = output.detach()
        return hook

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()

    def get_resized(self, target_size=(256, 256)):
        result = {}
        for name, att in self.maps.items():
            arr = F.interpolate(
                att.float(), size=target_size,
                mode="bilinear", align_corners=False
            ).squeeze().cpu().numpy()
            arr -= arr.min()
            if arr.max() > 0:
                arr /= arr.max()
            result[name] = arr
        return result


def preprocess(image_bgr, device):
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img_rgb, (256, 256)).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device)
    return tensor, img_rgb


def apply_heatmap(image_rgb, cam, alpha=0.55):
    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    img_uint8 = (image_rgb / 255.0 * 255).astype(np.uint8) if image_rgb.max() <= 1 else image_rgb.astype(np.uint8)
    img_uint8 = cv2.resize(img_uint8, (256, 256))
    blended = cv2.addWeighted(img_uint8, 1 - alpha, heatmap_rgb, alpha, 0)
    return blended


def load_models(device, unet_path="unet_flood_model.pth", attn_path="best_model_attention.pth"):
    unet = UNet().to(device)
    unet.load_state_dict(torch.load(unet_path, map_location=device))
    unet.eval()
    attn = AttentionUNet().to(device)
    attn.load_state_dict(torch.load(attn_path, map_location=device))
    attn.eval()
    return unet, attn


def run_gradcam(
    image_path,
    model_type="both",
    unet_path="unet_flood_model.pth",
    attn_path="best_model_attention.pth",
    save=False,
    save_path="gradcam_result.png",
    show_attention_maps=True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    img_tensor, img_rgb = preprocess(image_bgr, device)
    img_uint8 = img_rgb.astype(np.uint8)

    unet, attn_unet = load_models(device, unet_path, attn_path)

    results = {}

    if model_type in ("unet", "both"):
        print("Computing Grad-CAM for UNet...")
        gc_unet = GradCAM(unet, unet.bottleneck.conv, device)
        cam_unet = gc_unet.generate(img_tensor.clone())
        gc_unet.remove_hooks()
        results["unet"] = cam_unet
        print("UNet Grad-CAM done ✓")

    att_maps = {}
    if model_type in ("attention", "both"):
        print("Computing Grad-CAM for AttentionUNet...")
        gc_attn = GradCAM(attn_unet, attn_unet.bottleneck.conv, device)
        att_extractor = AttentionMapExtractor(attn_unet) if show_attention_maps else None
        cam_attn = gc_attn.generate(img_tensor.clone())
        gc_attn.remove_hooks()
        results["attention"] = cam_attn
        if att_extractor:
            att_maps = att_extractor.get_resized()
            att_extractor.remove_hooks()
        print("AttentionUNet Grad-CAM done ✓")

    if model_type == "both":
        _plot_both(img_uint8, results, att_maps, save, save_path)
    elif model_type == "unet":
        _plot_single(img_uint8, results["unet"], "U-Net", {}, save, save_path)
    else:
        _plot_single(img_uint8, results["attention"], "Attention U-Net", att_maps, save, save_path)

    return results


def _plot_both(img, results, att_maps, save, save_path):
    has_att = bool(att_maps)
    n_rows = 3 if has_att else 2
    fig = plt.figure(figsize=(16, 5 * n_rows))
    fig.patch.set_facecolor("#0f0f12")

    def _title(ax, txt, sub=""):
        ax.set_title(txt, color="white", fontsize=11, fontweight="bold", pad=6)
        if sub:
            ax.set_xlabel(sub, color="#aaa", fontsize=9)
        ax.axis("off")

    # Row 1 - UNet
    cam_unet = results["unet"]
    overlay_unet = apply_heatmap(img, cam_unet)

    ax = fig.add_subplot(n_rows, 4, 1)
    ax.imshow(img); _title(ax, "Input Image")

    ax = fig.add_subplot(n_rows, 4, 2)
    ax.imshow(cam_unet, cmap="jet"); _title(ax, "U-Net Grad-CAM")

    ax = fig.add_subplot(n_rows, 4, 3)
    ax.imshow(overlay_unet); _title(ax, "U-Net Overlay")

    ax = fig.add_subplot(n_rows, 4, 4)
    im = ax.imshow(cam_unet, cmap="jet")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _title(ax, "U-Net Colorbar")

    # Row 2 - Attention UNet
    cam_attn = results["attention"]
    overlay_attn = apply_heatmap(img, cam_attn)

    ax = fig.add_subplot(n_rows, 4, 5)
    ax.imshow(img); _title(ax, "Input Image")

    ax = fig.add_subplot(n_rows, 4, 6)
    ax.imshow(cam_attn, cmap="jet"); _title(ax, "Attention U-Net Grad-CAM")

    ax = fig.add_subplot(n_rows, 4, 7)
    ax.imshow(overlay_attn); _title(ax, "Attention U-Net Overlay")

    ax = fig.add_subplot(n_rows, 4, 8)
    diff = cam_attn.astype(float) - cam_unet.astype(float)
    vmax = max(abs(diff.min()), abs(diff.max())) + 1e-9
    ax.imshow(diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    _title(ax, "Difference (Attn - UNet)", "red=attn higher | blue=attn lower")

    # Row 3 - Attention gates
    if has_att:
        labels = {
            "att1": "Gate 1 (deep 64x64)",
            "att2": "Gate 2 (mid 128x128)",
            "att3": "Gate 3 (shallow 256x256)"
        }
        for col_i, (k, arr) in enumerate(sorted(att_maps.items())):
            ax = fig.add_subplot(n_rows, 4, 9 + col_i)
            ax.imshow(arr, cmap="viridis")
            _title(ax, labels.get(k, k))

        composite = np.mean(list(att_maps.values()), axis=0)
        ax = fig.add_subplot(n_rows, 4, 12)
        ax.imshow(composite, cmap="viridis")
        _title(ax, "Composite (mean of gates)")

    fig.suptitle("Grad-CAM · Flood Segmentation", color="white", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved → {save_path}")
    plt.show()


def _plot_single(img, cam, model_name, att_maps, save, save_path):
    has_att = bool(att_maps)
    n_rows = 2 if has_att else 1
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 5 * n_rows))
    fig.patch.set_facecolor("#0f0f12")
    if n_rows == 1:
        axes = [axes]

    overlay = apply_heatmap(img, cam)

    def _title(ax, txt, sub=""):
        ax.set_title(txt, color="white", fontsize=11, fontweight="bold", pad=6)
        if sub:
            ax.set_xlabel(sub, color="#aaa", fontsize=9)
        ax.axis("off")

    axes[0][0].imshow(img);              _title(axes[0][0], "Input Image")
    axes[0][1].imshow(cam, cmap="jet");  _title(axes[0][1], f"{model_name} Grad-CAM")
    axes[0][2].imshow(overlay);          _title(axes[0][2], f"{model_name} Overlay")
    im = axes[0][3].imshow(cam, cmap="jet")
    plt.colorbar(im, ax=axes[0][3], fraction=0.046, pad=0.04)
    _title(axes[0][3], "Colorbar")

    if has_att:
        labels = {"att1": "Gate 1 (64x64)", "att2": "Gate 2 (128x128)", "att3": "Gate 3 (256x256)"}
        for i, (k, arr) in enumerate(sorted(att_maps.items())):
            axes[1][i].imshow(arr, cmap="viridis")
            _title(axes[1][i], labels.get(k, k))
        composite = np.mean(list(att_maps.values()), axis=0)
        axes[1][3].imshow(composite, cmap="viridis")
        _title(axes[1][3], "Composite gates")

    fig.suptitle(f"Grad-CAM · {model_name}", color="white", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved → {save_path}")
    plt.show()


def run_gradcam_on_batch(
    image_paths,
    model_type="both",
    unet_path="unet_flood_model.pth",
    attn_path="best_model_attention.pth",
    save_dir="gradcam_outputs",
    show_attention_maps=True,
):
    import os
    os.makedirs(save_dir, exist_ok=True)
    for i, path in enumerate(image_paths):
        print(f"\n[{i+1}/{len(image_paths)}] {path}")
        fname = os.path.splitext(os.path.basename(path))[0]
        save_path = os.path.join(save_dir, f"{fname}_gradcam.png")
        try:
            run_gradcam(
                path,
                model_type=model_type,
                unet_path=unet_path,
                attn_path=attn_path,
                save=True,
                save_path=save_path,
                show_attention_maps=show_attention_maps,
            )
        except Exception as e:
            print(f"  Failed: {e}")