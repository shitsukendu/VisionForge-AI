import os
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from PIL import Image
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    CLIPProcessor,
    CLIPModel
)

import torch


# =========================================================
# ENVIRONMENT
# =========================================================

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    st.error("HF_TOKEN not found in .env file.")
    st.stop()


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="VisionForge AI",
    page_icon="🎨",
    layout="wide"
)


# =========================================================
# HUGGING FACE CLIENT
# =========================================================

client = InferenceClient(
    provider="auto",
    api_key=HF_TOKEN
)


BLIP_MODEL = "Salesforce/blip-image-captioning-base"


@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained(BLIP_MODEL)
    model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL)
    model.eval()
    return processor, model


blip_processor, blip_model = load_blip_model()


# =========================
# CLIP VISION MODEL
# =========================

CLIP_MODEL = "openai/clip-vit-base-patch32"


@st.cache_resource
def load_clip_model():

    processor = CLIPProcessor.from_pretrained(
        CLIP_MODEL
    )

    model = CLIPModel.from_pretrained(
        CLIP_MODEL
    )

    model.eval()

    return processor, model


clip_processor, clip_model = load_clip_model()


# =========================================================
# MODEL
# =========================================================

TEXT_TO_IMAGE_MODEL = (
    "black-forest-labs/FLUX.1-schnell"
)


# =========================================================
# SESSION STATE
# =========================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "reuse_settings" not in st.session_state:
    st.session_state.reuse_settings = None


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* =====================================================
       MAIN APP BACKGROUND
       ===================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 85% 5%,
                rgba(25, 100, 220, 0.20),
                transparent 32%
            ),
            radial-gradient(
                circle at 15% 80%,
                rgba(35, 80, 180, 0.10),
                transparent 35%
            ),
            linear-gradient(
                135deg,
                #050b16 0%,
                #081426 45%,
                #0b1930 100%
            );

        color: #f5f7ff;
    }


    /* =====================================================
       STREAMLIT TOP BAR
       ===================================================== */

    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    div[data-testid="stToolbar"] {
        color: white !important;
    }


    /* =====================================================
       MAIN CONTENT
       ===================================================== */

    .main .block-container {
        max-width: 1250px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }


    /* =====================================================
       SIDEBAR
       ===================================================== */

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #050b15 0%,
                #071426 50%,
                #040a13 100%
            );

        border-right:
            1px solid
            rgba(70, 150, 255, 0.18);
    }

    section[data-testid="stSidebar"] * {
        color: #edf4ff;
    }


    /* =====================================================
       MAIN TITLE
       ===================================================== */

    .main-title {
        display: flex;
        align-items: center;

        font-size: 48px;
        font-weight: 800;

        letter-spacing: -1.5px;

        margin-bottom: 4px;
    }


    /* =====================================================
       TITLE ICON
       ===================================================== */

    .title-icon {
        display: inline-flex;

        width: 46px;
        height: 46px;

        align-items: center;
        justify-content: center;

        margin-right: 12px;

        border-radius: 13px;

        background:
            linear-gradient(
                135deg,
                #42ddff 0%,
                #378cff 55%,
                #8b5cf6 100%
            );

        color: #ffffff;

        font-size: 25px;
        font-weight: 800;

        box-shadow:
            0 0 22px
            rgba(50, 170, 255, 0.25);
    }

    .title-icon::before {
        content: "✦";
    }


    /* =====================================================
       TITLE TEXT
       ===================================================== */

    .title-text {
        background:
            linear-gradient(
                90deg,
                #ffffff 0%,
                #d9fbff 20%,
                #55ddff 42%,
                #269dff 65%,
                #5966f2 82%,
                #9b5cff 100%
            );

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;

        background-clip: text;

        filter:
            drop-shadow(
                0 0 15px
                rgba(50, 180, 255, 0.12)
            );
    }


    /* =====================================================
       SUBTITLE
       ===================================================== */

    .subtitle {
        font-size: 17px;
        color: #91a8c8;
        margin-bottom: 30px;
    }


    /* =====================================================
       HEADINGS
       ===================================================== */

    h1,
    h2,
    h3,
    h4 {
        color: #f5f8ff !important;
    }


    /* =====================================================
       TEXT
       ===================================================== */

    p,
    label {
        color: #dce8f8;
    }


    /* =====================================================
       TEXT INPUTS
       ===================================================== */

    textarea,
    input {
        background-color: #101f38 !important;

        color: #f4f8ff !important;

        border:
            1px solid
            rgba(85, 155, 255, 0.28) !important;

        border-radius: 10px !important;
    }


    textarea::placeholder,
    input::placeholder {
        color: #7185a3 !important;
    }


    /* =====================================================
       SELECT BOX
       ===================================================== */

    div[data-baseweb="select"] > div {
        background-color: #101f38 !important;

        color: #f4f8ff !important;

        border:
            1px solid
            rgba(85, 155, 255, 0.28) !important;

        border-radius: 10px !important;
    }


    /* =====================================================
       BUTTONS
       ===================================================== */

    .stButton > button {
        width: 100%;

        min-height: 42px;

        border-radius: 10px;

        font-weight: 650;

        color: #ffffff !important;

        background:
            linear-gradient(
                90deg,
                #10baf5 0%,
                #397cff 55%,
                #8b4dff 100%
            );

        border:
            1px solid
            rgba(90, 170, 255, 0.25);

        box-shadow:
            0 6px 20px
            rgba(30, 120, 255, 0.16);
    }


    .stButton > button:hover {
        transform: translateY(-1px);

        box-shadow:
            0 8px 25px
            rgba(30, 140, 255, 0.28);
    }


    /* =====================================================
       DOWNLOAD BUTTON
       ===================================================== */

    .stDownloadButton > button {
        width: 100%;

        border-radius: 10px;

        background: #102844;

        color: #e5f7ff !important;

        border:
            1px solid
            rgba(75, 165, 255, 0.28);
    }


    .stDownloadButton > button:hover {
        background: #15385d;

        border-color: #38bdf8;
    }


    /* =====================================================
       EXPANDER
       ===================================================== */

    div[data-testid="stExpander"] {
        background:
            rgba(9, 27, 49, 0.82);

        border:
            1px solid
            rgba(75, 155, 255, 0.18);

        border-radius: 11px;
    }


    /* =====================================================
       ALERTS
       ===================================================== */

    div[data-testid="stAlert"] {
        border-radius: 10px;
    }


    /* =====================================================
       DIVIDER
       ===================================================== */

    hr {
        border-color:
            rgba(100, 165, 255, 0.12);
    }


    /* =====================================================
       IMAGES
       ===================================================== */

    img {
        border-radius: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    # =====================================================
    # ADVANCED SETTINGS
    # =====================================================

    st.markdown(
        "## ⚙️ Advanced Settings"
    )

    # -----------------------------------------------------
    # INFERENCE STEPS
    # -----------------------------------------------------

    steps = st.number_input(
        "Inference Steps",
        min_value=1,
        max_value=8,
        value=4,
        step=1,
        help=(
            "Higher values can improve generation "
            "quality but may take longer."
        )
    )

    # -----------------------------------------------------
    # SEED
    # -----------------------------------------------------

    seed_option = st.selectbox(
        "Seed",
        [
            "Random",
            "Fixed"
        ],
        index=0
    )

    if seed_option == "Fixed":

        seed = st.number_input(
            "Seed Value",
            min_value=0,
            max_value=999999999,
            value=42,
            step=1
        )

    else:

        seed = None

    st.divider()

    # =====================================================
    # GENERATION HISTORY
    # =====================================================

    with st.expander(
        "🕘 Generation History",
        expanded=False
    ):

        if not st.session_state.history:

            st.info(
                "No generations yet."
            )

        else:

            for i, item in enumerate(
                reversed(
                    st.session_state.history
                )
            ):

                history_index = (
                    len(
                        st.session_state.history
                    ) - 1 - i
                )

                # -----------------------------------------
                # IMAGE
                # -----------------------------------------

                st.image(
                    item["image"],
                    use_container_width=True
                )

                # -----------------------------------------
                # PROMPT
                # -----------------------------------------

                prompt_preview = (
                    item["prompt"]
                )

                if len(prompt_preview) > 80:

                    prompt_preview = (
                        prompt_preview[:80]
                        + "..."
                    )

                st.caption(
                    f"✍️ {prompt_preview}"
                )

                # -----------------------------------------
                # RESOLUTION
                # -----------------------------------------

                st.caption(
                    f"📐 {item['width']} × "
                    f"{item['height']}"
                )

                # -----------------------------------------
                # STEPS
                # -----------------------------------------

                st.caption(
                    f"⚙️ Steps: "
                    f"{item['steps']}"
                )

                # -----------------------------------------
                # SEED
                # -----------------------------------------

                if item["seed"] is not None:

                    st.caption(
                        f"🎲 Seed: "
                        f"{item['seed']}"
                    )

                else:

                    st.caption(
                        "🎲 Seed: Random"
                    )

                # -----------------------------------------
                # NEGATIVE PROMPT
                # -----------------------------------------

                if (
                    item["negative_prompt"]
                    .strip()
                ):

                    with st.expander(
                        "🚫 Negative Prompt"
                    ):

                        st.write(
                            item[
                                "negative_prompt"
                            ]
                        )

                # -----------------------------------------
                # REUSE SETTINGS
                # -----------------------------------------

                if st.button(
                    "🔄 Reuse Settings",
                    key=f"reuse_{history_index}"
                ):

                    st.session_state.reuse_settings = {

                        "prompt":
                            item["prompt"],

                        "negative_prompt":
                            item[
                                "negative_prompt"
                            ],

                        "width":
                            item["width"],

                        "height":
                            item["height"],

                        "steps":
                            item["steps"],

                        "seed":
                            item["seed"]
                    }

                    st.rerun()

                # -----------------------------------------
                # DOWNLOAD
                # -----------------------------------------

                buffer = BytesIO()

                item["image"].save(
                    buffer,
                    format="PNG"
                )

                st.download_button(
                    label="⬇️ Download",
                    data=buffer.getvalue(),
                    file_name=(
                        f"visionforge_"
                        f"{history_index + 1}.png"
                    ),
                    mime="image/png",
                    key=(
                        f"history_download_"
                        f"{history_index}"
                    )
                )

                st.divider()


# =========================================================
# MAIN TITLE
# =========================================================

st.markdown(
    """
    <div class="main-title">
        <span class="title-icon"></span>
        <span class="title-text">
            VisionForge AI
        </span>
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown(
    '<div class="subtitle">'
    'AI-Powered Image Generation'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# REUSE SETTINGS
# =========================================================

reuse = (
    st.session_state.reuse_settings
)

if reuse:

    st.info(
        "🔄 Previous generation settings loaded."
    )

    default_prompt = reuse[
        "prompt"
    ]

    default_negative_prompt = (
        reuse["negative_prompt"]
    )

else:

    default_prompt = ""

    default_negative_prompt = ""


# =========================================
# INTELLIGENT PROMPT ASSISTANT - V6
# =========================================

st.markdown("### ✨ Intelligent Prompt Assistant")

prompt = st.text_area(
    "Enter your prompt",
    placeholder="Example: futuristic robot",
    height=120
)

prompt_assistant = st.checkbox(
    "🧠 Enhance my prompt automatically",
    value=False
)

style = st.selectbox(
    "🎨 Choose Image Style",
    [
        "Realistic",
        "Cinematic",
        "Digital Art",
        "Anime",
        "3D Render"
    ]
)

enhanced_prompt = prompt

if prompt_assistant and prompt.strip():

    base_prompt = prompt.strip().lower()

    if any(word in base_prompt for word in ["robot", "android", "machine"]):

        enhancement = (
            "futuristic robotic design, metallic surface details, "
            "advanced mechanical components, cinematic lighting, "
            "realistic materials, sharp focus, highly detailed, "
            "professional digital artwork, high quality"
        )

    elif any(word in base_prompt for word in ["cat", "dog", "animal", "bird"]):

        enhancement = (
            "realistic natural details, detailed fur and texture, "
            "expressive appearance, natural lighting, sharp focus, "
            "cinematic composition, highly detailed, high quality"
        )

    elif any(word in base_prompt for word in ["car", "vehicle", "bike", "motorcycle"]):

        enhancement = (
            "premium vehicle design, realistic reflections, "
            "detailed surface materials, cinematic lighting, "
            "dynamic composition, sharp focus, photorealistic, high quality"
        )

    elif any(word in base_prompt for word in ["shoe", "sneaker"]):

        enhancement = (
            "detailed product photography, realistic materials, "
            "professional studio lighting, clean composition, "
            "sharp focus, realistic texture, high quality"
        )

    elif any(word in base_prompt for word in ["house", "building", "architecture"]):

        enhancement = (
            "detailed architectural design, realistic textures, "
            "cinematic environment, professional lighting, "
            "sharp details, realistic materials, high quality"
        )

    elif any(word in base_prompt for word in ["person", "man", "woman", "human"]):

        enhancement = (
            "detailed human features, natural lighting, "
            "realistic skin texture, cinematic composition, "
            "sharp focus, professional photography, high quality"
        )

    else:

        enhancement = (
            "highly detailed, cinematic composition, "
            "professional lighting, realistic textures, "
            "sharp focus, visually appealing, "
            "high quality"
        )

    style_prompts = {
    "Realistic": "photorealistic, natural lighting, realistic textures",
    "Cinematic": "cinematic lighting, dramatic atmosphere, movie scene composition",
    "Digital Art": "professional digital artwork, detailed artistic rendering",
    "Anime": "anime style, vibrant colors, detailed anime artwork",
    "3D Render": "high quality 3D render, realistic materials, detailed 3D modeling"
    }

    style_enhancement = style_prompts[style]

    enhanced_prompt = (
        f"{prompt.strip()}, "
        f"{enhancement}, "
        f"{style_enhancement}"
    )

    st.markdown("### 🚀 Enhanced Prompt")
    st.info(enhanced_prompt)

    st.markdown("### 🔍 Prompt Preview")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📝 Original Prompt**")
        st.info(prompt.strip())

    with col2:
        st.markdown("**✨ Enhanced Prompt**")
        st.success(enhanced_prompt)

# =========================================================
# GENERATION SETTINGS
# =========================================================

col1, col2 = st.columns(2)


with col1:

    st.markdown(
        "##### 📐 Aspect Ratio"
    )

    if reuse:

        if (
            reuse["width"] == 1024
            and reuse["height"] == 1024
        ):

            default_ratio = 0

        elif (
            reuse["width"] == 1360
            and reuse["height"] == 768
        ):

            default_ratio = 1

        else:

            default_ratio = 2

    else:

        default_ratio = 0

    aspect_ratio = st.selectbox(
        "Aspect Ratio",
        [
            "Square (1024 × 1024)",
            "Landscape (1360 × 768)",
            "Portrait (768 × 1360)"
        ],
        index=default_ratio,
        label_visibility="collapsed"
    )


with col2:

    st.markdown(
        "##### 🚫 Negative Prompt"
    )

    negative_prompt = st.text_area(
        "Negative Prompt",
        value=default_negative_prompt,
        placeholder=(
            "Things you don't want in the image..."
        ),
        height=100,
        label_visibility="collapsed"
    )

    if prompt_assistant:
        smart_negative_prompt = (
            "blurry, low quality, distorted, deformed, "
            "extra limbs, duplicate objects, bad anatomy, "
            "oversaturated, noisy, watermark, text, "
            "cropped, out of frame"
        )

        if not negative_prompt.strip():
            negative_prompt = smart_negative_prompt


# =========================================================
# ASPECT RATIO
# =========================================================

if (
    aspect_ratio
    == "Square (1024 × 1024)"
):

    width = 1024
    height = 1024

elif (
    aspect_ratio
    == "Landscape (1360 × 768)"
):

    width = 1360
    height = 768

else:

    width = 768
    height = 1360


# =========================================================
# GENERATE BUTTON
# =========================================================

generate = st.button(
    "✨ Generate Image",
    type="primary"
)


# =========================
# IMAGE ANALYSIS V5
# =========================

st.markdown("---")
st.subheader("🔍 Image Analysis")

uploaded_image = st.file_uploader(
    "Upload an image to analyze",
    type=["jpg", "jpeg", "png"],
    key="image_analysis_upload"
)

if uploaded_image is not None:

    image = Image.open(uploaded_image).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    analyze_image = st.button(
        "🔍 Analyze Image",
        use_container_width=True
    )

    if analyze_image:

        with st.spinner("Understanding image..."):

            # =========================================
            # AI DESCRIPTION
            # =========================================

            inputs = blip_processor(
                images=image,
                return_tensors="pt"
            )

            output = blip_model.generate(
                **inputs,
                max_new_tokens=60
            )

            description = blip_processor.decode(
                output[0],
                skip_special_tokens=True
            )

            # =========================================
            # MAIN SUBJECT
            # =========================================

            subject_inputs = blip_processor(
                images=image,
                text="a photo of",
                return_tensors="pt"
            )

            subject_output = blip_model.generate(
                **subject_inputs,
                max_new_tokens=60
            )

            subject = blip_processor.decode(
                subject_output[0],
                skip_special_tokens=True
            )

            # =========================================
            # SMART VISUAL CLASSIFICATION
            # =========================================

            visual_categories = [
                "robot",
                "person",
                "animal",
                "car",
                "truck",
                "motorcycle",
                "bicycle",
                "airplane",
                "boat",
                "computer",
                "laptop",
                "phone",
                "headphones",
                "shoe",
                "bottle",
                "chair",
                "table",
                "bed",
                "building",
                "food",
                "nature",
                "electronic device",
                "toy",
                "clothing"
            ]

            text_prompts = [
                f"a photo of a {category}"
                for category in visual_categories
            ]

            clip_inputs = clip_processor(
                text=text_prompts,
                images=image,
                return_tensors="pt",
                padding=True
            )

            with torch.no_grad():

                clip_outputs = clip_model(
                    **clip_inputs
                )

            logits = clip_outputs.logits_per_image[0]

            probabilities = logits.softmax(
                dim=0
            )

            top_results = torch.topk(
                probabilities,
                k=5
            )

            smart_results = []

            for score, index in zip(
                top_results.values,
                top_results.indices
            ):

                category = visual_categories[
                    index.item()
                ]

                confidence = score.item() * 100

                smart_results.append(
                    (category, confidence)
                )

            # =========================================
            # DOMINANT COLORS
            # =========================================

            small_image = image.resize(
                (100, 100)
            )

            quantized_image = small_image.quantize(
                colors=5
            ).convert("RGB")

            color_counts = quantized_image.getcolors(
                maxcolors=10000
            )

            color_counts = sorted(
                color_counts,
                reverse=True
            )

            dominant_colors = []

            for count, rgb in color_counts[:5]:

                r, g, b = rgb

                hex_color = (
                    f"#{r:02X}{g:02X}{b:02X}"
                )

                dominant_colors.append(
                    (hex_color, rgb)
                )

        st.success(
            "✅ Image analysis completed!"
        )

        # =========================================
        # AI DESCRIPTION
        # =========================================

        st.markdown("### 📝 AI Description")

        st.write(description)

        # =========================================
        # MAIN SUBJECT
        # =========================================

        st.markdown("### 🎯 Main Subject")

        st.write(subject)

        # =========================================
        # SMART VISUAL UNDERSTANDING
        # =========================================

        st.markdown(
            "### 🧠 Smart Visual Understanding"
        )

        for category, confidence in smart_results:

            st.write(
                f"• **{category.title()}** — "
                f"{confidence:.1f}%"
            )

        st.caption(
            "Scores represent relative similarity "
            "between the image and the selected "
            "visual concepts."
        )

        # =========================================
        # DOMINANT COLORS
        # =========================================

        st.markdown("### 🎨 Dominant Colors")

        color_columns = st.columns(
            len(dominant_colors)
        )

        for col, (hex_color, rgb) in zip(
            color_columns,
            dominant_colors
        ):

            with col:

                st.markdown(
                    f"""
                    <div style="
                        background-color:{hex_color};
                        width:70px;
                        height:50px;
                        border-radius:10px;
                        border:1px solid #888;
                        margin-bottom:5px;
                    "></div>
                    """,
                    unsafe_allow_html=True
                )

                st.caption(
                    f"{hex_color}\nRGB {rgb}"
                )

        # =========================================
        # IMAGE INFORMATION
        # =========================================

        st.markdown(
            "### 📊 Image Information"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.write(
                "**Width:**",
                f"{image.width} px"
            )

            st.write(
                "**Height:**",
                f"{image.height} px"
            )

        with col2:

            st.write(
                "**Color Mode:**",
                image.mode
            )

            st.write(
                "**Format:**",
                image.format or "Uploaded Image"
            )

        # =========================================
        # AI INSIGHTS
        # =========================================

        st.markdown("### 💡 AI Insights")

        top_category = smart_results[0][0]

        st.info(
            f"The visual model considers "
            f"**{top_category}** the strongest "
            f"matching concept among the available "
            f"categories."
        )


# =========================================================
# TEXT TO IMAGE
# =========================================================

if generate:

    if not prompt.strip():

        st.warning(
            "⚠️ Please enter a prompt first."
        )

    else:

        with st.spinner(
            "🎨 Generating your image..."
        ):

            try:

                generation_params = {

                    "model":
                        TEXT_TO_IMAGE_MODEL,

                    "width":
                        width,

                    "height":
                        height,

                    "num_inference_steps":
                        steps
                }

                # -----------------------------------------
                # SEED
                # -----------------------------------------

                if seed is not None:

                    generation_params[
                        "seed"
                    ] = seed

                # -----------------------------------------
                # NEGATIVE PROMPT
                # -----------------------------------------

                if negative_prompt.strip():

                    generation_params[
                        "negative_prompt"
                    ] = (
                        negative_prompt.strip()
                    )

                # -----------------------------------------
                # GENERATE IMAGE
                # -----------------------------------------

                image = client.text_to_image(
                    enhanced_prompt,
                    **generation_params
                )

                # -----------------------------------------
                # HISTORY
                # -----------------------------------------

                st.session_state.history.append(
                    {
                        "image":
                            image,

                        "prompt":
                            prompt,

                        "negative_prompt":
                            negative_prompt,

                        "width":
                            width,

                        "height":
                            height,

                        "steps":
                            steps,

                        "seed":
                            seed
                    }
                )

                # -----------------------------------------
                # CLEAR REUSE
                # -----------------------------------------

                st.session_state[
                    "reuse_settings"
                ] = None

                # -----------------------------------------
                # SUCCESS
                # -----------------------------------------

                st.success(
                    "✅ Image generated successfully!"
                )

                # -----------------------------------------
                # DISPLAY
                # -----------------------------------------

                st.image(
                    image,
                    caption=(
                        f"{width} × {height}"
                    ),
                    use_container_width=True
                )

                # -----------------------------------------
                # DOWNLOAD
                # -----------------------------------------

                buffer = BytesIO()

                image.save(
                    buffer,
                    format="PNG"
                )

                st.download_button(
                    label="⬇️ Download Image",
                    data=buffer.getvalue(),
                    file_name=(
                        "visionforge_generated.png"
                    ),
                    mime="image/png"
                )

            except Exception as e:
                error_message = str(e)

                if "402" in error_message or "Payment Required" in error_message:
                    st.warning(
                        "⚠️ Image generation service is temporarily unavailable "
                        "because the free inference limit has been reached. "
                        "Please try again later."
                    )
                else:
                    st.error(
                        "❌ Image generation failed:\n\n" + error_message
                    )
