import torch
import cv2
import numpy as np
from attention_unet_model import AttentionUNet
from unet_flood_model import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# LOAD MODELS
# -----------------------------
unet_model = UNet()
unet_model.load_state_dict(
    torch.load("unet_flood_model.pth", map_location=device)
)
unet_model.to(device).eval()

attn_model = AttentionUNet()
attn_model.load_state_dict(
    torch.load("best_model_attention.pth", map_location=device)
)
attn_model.to(device).eval()


# -----------------------------
# PREPROCESS FUNCTION
# -----------------------------
def preprocess(image):
    """
    Prepare image for model
    """
    image = cv2.resize(image, (256, 256))

    image = image.astype(np.float32) / 255.0

    # If you used normalization during training, uncomment:
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]
    # image = (image - mean) / std

    image = np.transpose(image, (2, 0, 1))  # HWC -> CHW

    image = torch.from_numpy(image).unsqueeze(0).to(device)

    return image


# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict_image(image, model_type="attention"):

    if image is None:
        raise ValueError("Input image is None")

    # Select model
    if model_type == "unet":
        model = unet_model
    elif model_type == "attention":
        model = attn_model
    else:
        raise ValueError("model_type must be 'unet' or 'attention'")

    # Convert BGR -> RGB (important for consistency)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    original_size = (image.shape[1], image.shape[0])

    # Preprocess
    img_tensor = preprocess(image)

    # Inference
    with torch.no_grad():
        output = model(img_tensor)
        output = torch.sigmoid(output)

    # Convert to numpy safely
    pred = output.squeeze().cpu().numpy()

    # -----------------------------
    # RESIZE BACK (FIXED)
    # -----------------------------
    pred = cv2.resize(
        pred,
        original_size,
        interpolation=cv2.INTER_LINEAR  # smooth for probability
    )

    # Ensure values are valid
    pred = np.clip(pred, 0, 1)

    return pred