import streamlit as st
import cv2
import numpy as np
from PIL import Image
from predict import predict_image, unet_model, attn_model, device
from streamlit_image_comparison import image_comparison
from gradcam_flood import GradCAM, AttentionMapExtractor, apply_heatmap
import torch

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Flood Segmentation Comparison",
    layout="wide"
)

# -------------------------
# STYLE
# -------------------------
st.markdown("""
<style>
.main {padding-top: 2rem;}
.block-container {padding-top: 1rem;}
</style>
""", unsafe_allow_html=True)

# -------------------------
# TITLE
# -------------------------
st.title("🌊 U-Net vs Attention U-Net Flood Segmentation")
st.caption("Compare segmentation performance on drone imagery")

st.markdown("---")

# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.header("⚙️ Settings")

unet_threshold = st.sidebar.slider("U-Net Threshold", 0.1, 0.9, 0.6, 0.05)
attn_threshold = st.sidebar.slider("Attention U-Net Threshold", 0.90, 0.999, 0.98, 0.001)

clean_mask   = st.sidebar.checkbox("🧹 Remove Noise (Recommended)", True)
show_prob    = st.sidebar.checkbox("🧠 Show Probability Maps (Debug)", False)
show_gradcam = st.sidebar.checkbox("🔥 Show Grad-CAM Explainability", False)
show_gates   = st.sidebar.checkbox("🔬 Show Attention Gate Maps", False)

st.sidebar.markdown("### 🎯 Note")
st.sidebar.info(
    "Different models require different thresholds due to probability calibration."
)

# -------------------------
# HELPER FUNCTIONS
# -------------------------
def post_process(mask):
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def run_gradcam_streamlit(image_bgr, show_attention_gates=False):
    """
    Runs Grad-CAM reusing already-loaded models from predict.py.
    No double loading — no extra memory usage.
    """
    # Reuse models already loaded in predict.py
    unet, attn_unet = unet_model, attn_model

    # Preprocess
    img_rgb   = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img       = cv2.resize(img_rgb, (256, 256)).astype(np.float32) / 255.0
    tensor    = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device)
    img_uint8 = (img * 255).astype(np.uint8)

    results = {}

    # U-Net Grad-CAM
    gc_unet  = GradCAM(unet, unet.bottleneck.conv, device)
    cam_unet = gc_unet.generate(tensor.clone())
    gc_unet.remove_hooks()
    results["unet_cam"]     = cam_unet
    results["unet_overlay"] = apply_heatmap(img_uint8, cam_unet)

    # Attention UNet Grad-CAM
    att_extractor = AttentionMapExtractor(attn_unet) if show_attention_gates else None
    gc_attn  = GradCAM(attn_unet, attn_unet.bottleneck.conv, device)
    cam_attn = gc_attn.generate(tensor.clone())
    gc_attn.remove_hooks()
    results["attn_cam"]     = cam_attn
    results["attn_overlay"] = apply_heatmap(img_uint8, cam_attn)

    # Difference map
    diff       = cam_attn.astype(float) - cam_unet.astype(float)
    diff_norm  = ((diff + 1) / 2 * 255).astype(np.uint8)
    diff_color = cv2.applyColorMap(diff_norm, cv2.COLORMAP_COOL)
    results["diff"] = cv2.cvtColor(diff_color, cv2.COLOR_BGR2RGB)

    # Attention gates
    if att_extractor:
        att_maps = att_extractor.get_resized()
        att_extractor.remove_hooks()
        results["att_maps"] = att_maps

    return results


# -------------------------
# UPLOAD IMAGE
# -------------------------
uploaded_file = st.file_uploader(
    "📤 Upload Drone Image",
    type=["jpg", "jpeg", "png"]
)

# -------------------------
# MAIN
# -------------------------
if uploaded_file is not None:
    try:
        image_pil = Image.open(uploaded_file).convert("RGB")
        image     = np.array(image_pil)
        image_cv  = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Predictions
        with st.spinner("🔍 Running U-Net..."):
            unet_prob = predict_image(image_cv, model_type="unet")

        with st.spinner("🔍 Running Attention U-Net..."):
            attn_prob = predict_image(image_cv, model_type="attention")

        # Threshold
        unet_mask = (unet_prob >= unet_threshold).astype(np.uint8)
        attn_mask = (attn_prob >= attn_threshold).astype(np.uint8)

        if clean_mask:
            unet_mask = post_process(unet_mask)
            attn_mask = post_process(attn_mask)

        st.success("✅ Both models completed successfully")

        # Flood Coverage
        unet_flood = np.mean(unet_mask) * 100
        attn_flood = np.mean(attn_mask) * 100

        st.markdown("### 📊 Flood Coverage")
        col1, col2 = st.columns(2)
        col1.metric("U-Net", f"{unet_flood:.2f}%")
        col2.metric("Attention U-Net", f"{attn_flood:.2f}%")

        # Model Outputs
        st.markdown("### 🧠 Model Outputs")
        col1, col2, col3 = st.columns(3)
        col1.image(image,           caption="Input",          use_container_width=True)
        col2.image(unet_mask * 255, caption="U-Net Mask",     use_container_width=True)
        col3.image(attn_mask * 255, caption="Attention Mask", use_container_width=True)

        # Overlay
        def create_overlay(img, mask):
            colored = np.zeros_like(img)
            colored[:, :, 1] = mask * 255
            overlay = cv2.addWeighted(img, 0.7, colored, 0.3, 0)
            return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        unet_overlay = create_overlay(image_cv, unet_mask)
        attn_overlay = create_overlay(image_cv, attn_mask)

        st.markdown("### 🛰️ Overlay Comparison")
        col1, col2 = st.columns(2)
        col1.image(unet_overlay, caption="U-Net Overlay",     use_container_width=True)
        col2.image(attn_overlay, caption="Attention Overlay", use_container_width=True)

        # Slider Comparison
        st.markdown("### 🔄 Before vs Attention")
        image_comparison(
            img1=image,
            img2=attn_overlay,
            label1="Original",
            label2="Attention U-Net",
            width=900
        )

        # ── GRAD-CAM SECTION ──────────────────────────────────────
        if show_gradcam:
            st.markdown("---")
            st.markdown("### 🔥 Grad-CAM Explainability")
            st.caption("Heatmap shows which regions the model focused on to make its prediction")

            with st.spinner("🔥 Computing Grad-CAM heatmaps..."):
                gradcam_results = run_gradcam_streamlit(
                    image_cv,
                    show_attention_gates=show_gates
                )

            # Heatmaps row
            st.markdown("#### 🗺️ Grad-CAM Heatmaps")
            col1, col2, col3 = st.columns(3)
            col1.image(gradcam_results["unet_overlay"],
                       caption="🔵 U-Net Grad-CAM",
                       use_container_width=True)
            col2.image(gradcam_results["attn_overlay"],
                       caption="🔴 Attention U-Net Grad-CAM",
                       use_container_width=True)
            col3.image(gradcam_results["diff"],
                       caption="⚖️ Difference (Attn − UNet)",
                       use_container_width=True)

            st.info(
                "🔴 Red/Yellow = High activation (model focused here)  |  "
                "🔵 Blue = Low activation (model ignored here)  |  "
                "Difference map shows where Attention U-Net focuses differently from U-Net"
            )

            # Attention Gate Maps row
            if show_gates and "att_maps" in gradcam_results:
                st.markdown("#### 🔬 Attention Gate Maps")
                st.caption("Shows what each attention gate learned to focus on independently")

                att_maps = gradcam_results["att_maps"]
                labels   = {
                    "att1": "Gate 1 — Deep (64×64)",
                    "att2": "Gate 2 — Mid (128×128)",
                    "att3": "Gate 3 — Shallow (256×256)"
                }

                cols = st.columns(4)
                for i, (k, arr) in enumerate(sorted(att_maps.items())):
                    arr_color = cv2.applyColorMap(
                        (arr * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS
                    )
                    cols[i].image(
                        cv2.cvtColor(arr_color, cv2.COLOR_BGR2RGB),
                        caption=labels.get(k, k),
                        use_container_width=True
                    )

                # Composite gate
                composite  = np.mean(list(att_maps.values()), axis=0)
                comp_color = cv2.applyColorMap(
                    (composite * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS
                )
                cols[3].image(
                    cv2.cvtColor(comp_color, cv2.COLOR_BGR2RGB),
                    caption="Composite (mean of all gates)",
                    use_container_width=True
                )

                st.info(
                    "🟡 Yellow/Green = Gate is paying attention here  |  "
                    "🟣 Purple/Dark = Gate is ignoring this region"
                )

        # Download
        st.markdown("---")
        _, buffer = cv2.imencode(
            ".png",
            cv2.cvtColor(attn_overlay, cv2.COLOR_RGB2BGR)
        )
        st.download_button(
            "📥 Download Attention Result",
            buffer.tobytes(),
            "flood_result.png",
            "image/png"
        )

        st.markdown("---")
        st.caption("Built with Streamlit • PyTorch • Grad-CAM")

    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.exception(e)