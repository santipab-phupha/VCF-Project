import os
import sys
import shutil
import importlib.util
from io import BytesIO
from ultralytics import YOLO
from PIL import Image

import torch
# ─── FORCE CPU ONLY ─────────────────────────────────────────────────────────
torch.Tensor.cuda      = lambda self, *args, **kwargs: self
torch.nn.Module.cuda   = lambda self, *args, **kwargs: self
torch.cuda.synchronize = lambda *args, **kwargs: None
torch.cuda.is_available= lambda : False
torch.cuda.device_count= lambda : 0
_orig_to = torch.Tensor.to
def _to_cpu(self, *args, **kwargs):
    new_args = []
    for a in args:
        if isinstance(a, str) and a.lower().startswith("cuda"):
            new_args.append("cpu")
        elif isinstance(a, torch.device) and a.type=="cuda":
            new_args.append(torch.device("cpu"))
        else:
            new_args.append(a)
    if "device" in kwargs:
        dev = kwargs["device"]
        if (isinstance(dev, str) and dev.lower().startswith("cuda")) or \
           (isinstance(dev, torch.device) and dev.type=="cuda"):
            kwargs["device"] = torch.device("cpu")
    return _orig_to(self, *new_args, **kwargs)
torch.Tensor.to = _to_cpu

from torch.utils.data import DataLoader as _DL
def _dl0(ds, *a, **kw):
    kw['num_workers'] = 0
    return _DL(ds, *a, **kw)
import torch.utils.data as _du
_du.DataLoader = _dl0

import cv2
import numpy as np
import streamlit as st
from argparse import Namespace

# ─── DYNAMIC IMPORT ─────────────────────────────────────────────────────────
REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.append(REPO)
models_dir = os.path.join(REPO, "models")
os.makedirs(models_dir, exist_ok=True)
open(os.path.join(models_dir, "__init__.py"), "a").close()

def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m    = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    sys.modules[name] = m
    return m

dataset_mod = load_mod("dataset",     os.path.join(REPO, "dataset.py"))
decoder_mod = load_mod("decoder",     os.path.join(REPO, "decoder.py"))
draw_mod    = load_mod("draw_points", os.path.join(REPO, "draw_points.py"))
test_mod    = load_mod("test",        os.path.join(REPO, "test.py"))
load_mod("models.dec_net",     os.path.join(models_dir, "dec_net.py"))
load_mod("models.model_parts", os.path.join(models_dir, "model_parts.py"))
load_mod("models.resnet",      os.path.join(models_dir, "resnet.py"))
load_mod("models.spinal_net",  os.path.join(models_dir, "spinal_net.py"))

BaseDataset = dataset_mod.BaseDataset
Network     = test_mod.Network

# ─── STREAMLIT UI ───────────────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="Vertebral Compression Fracture")

st.markdown(
        """
    <div style='border: 2px solid #0080FF; border-radius: 5px; padding: 10px'>
        <h1 style='text-align: center; color: #0080FF'>
        🦴 Vertebral Compression Fracture Detection 🖼️
        </h1>
    </div>
        """, unsafe_allow_html=True)
st.markdown("")
st.markdown("") 
st.markdown("")
col1, col2, col3, col4 = st.columns(4)

with col4:
    feature = st.selectbox(
        "🔀 Select Feature",
        ["How to use", "AP - Detection", "AP - Cobb angle" , "LA - Image Segmetation", "Contract"],
        index=0,  # default to "AP"
        help="Choose which view to display"
    )

if feature == "How to use":
    st.markdown("## 📖 How to use this app")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div style='border:2px solid #00BFFF; border-radius:10px; padding:15px; text-align:center; background-color:#F0F8FF'>
                <h2>Step 1️⃣</h2>
                <p>Go to <b>AP - Detection</b> or <b>LA - Image Segmentation</b></p>
                <p>Select a sample image or upload your own image file.</p>
                <p style='color:#008000;'><b>✅ Tip:</b> Best with X-ray images with clear vertebra visibility.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div style='border:2px solid #00BFFF; border-radius:10px; padding:15px; text-align:center; background-color:#F0F8FF'>
                <h2>Step 2️⃣</h2>
                <p>Press the <b>Enter</b> button.</p>
                <p>The system will process your image automatically.</p>
                <p style='color:#FFA500;'><b>⏳ Note:</b> Processing time depends on image size.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div style='border:2px solid #00BFFF; border-radius:10px; padding:15px; text-align:center; background-color:#F0F8FF'>
                <h2>Step 3️⃣</h2>
                <p>See the prediction results:</p>
                <p style= text-align:left > 1. Bounding boxes & landmarks (AP)</p>
                <p style= text-align:left >  2. Segmentation masks (LA)</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(" ")
    st.info("สามารถเลือกฟีเจอร์ได้ผ่าน Select Feature โดยแต่ล่ะฟีเจอร์จะมีตัวอย่างกำกับให้ว่าเป็นยังไง")

# store original dimensions
elif feature == "AP - Detection":
    uploaded = st.file_uploader("", type=["jpg", "jpeg", "png"])
    orig_w = orig_h = None
    img0 = None
    run = st.button("Enter", use_container_width=True)
    # ─── Maintain selected sample in session state ─────────
    if "sample_img" not in st.session_state:
        st.session_state.sample_img = None

    # ─── SAMPLE BUTTONS ─────────────────────────────────────
    with col1:
        if st.button(" 1️⃣ Example",use_container_width=True):
            st.session_state.sample_img = "image_1.jpg"
    with col2:
        if st.button(" 2️⃣ Example",use_container_width=True):
            st.session_state.sample_img = "image_2.jpg"
    with col3:
        if st.button(" 3️⃣ Example",use_container_width=True):
            st.session_state.sample_img = "image_3.jpg"

    # ─── UI FOR UPLOAD + DISPLAY ───────────────────────────
    col4, col5, col6 = st.columns(3)
    with col4:
        st.subheader("1️⃣ Upload & Run")

        sample_img = st.session_state.sample_img  # read persisted choice

        # case 1: uploaded file
        if uploaded:
            buf = uploaded.getvalue()
            arr = np.frombuffer(buf, np.uint8)
            img0 = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            orig_h, orig_w = img0.shape[:2]
            st.image(cv2.cvtColor(img0, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_container_width=True)

        # case 2: selected sample image
        elif sample_img is not None:
            img_path = os.path.join(REPO, sample_img)
            img0 = cv2.imread(img_path)
            if img0 is not None:
                orig_h, orig_w = img0.shape[:2]
                st.image(cv2.cvtColor(img0, cv2.COLOR_BGR2RGB),
                         caption=f"Sample Image: {sample_img}",
                         use_container_width=True)
            else:
                st.error(f"Cannot find {sample_img} in directory!")



    with col5:
        st.subheader("2️⃣ Predictions")
    with col6:
        st.subheader("3️⃣ Heatmap")

    # ─── ARGS & CHECKPOINT ─────────────────────────────────
    args = Namespace(
        resume="model_30.pth",
        data_dir=os.path.join(REPO, "dataPath"),
        dataset="spinal",
        phase="test",
        input_h=1024,
        input_w=512,
        down_ratio=4,
        num_classes=1,
        K=17,
        conf_thresh=0.2,
    )
    weights_dir = os.path.join(REPO, "weights_spinal")
    os.makedirs(weights_dir, exist_ok=True)
    src_ckpt = os.path.join(REPO, "model_backup", args.resume)
    dst_ckpt = os.path.join(weights_dir, args.resume)
    if os.path.isfile(src_ckpt) and not os.path.isfile(dst_ckpt):
        shutil.copy(src_ckpt, dst_ckpt)

    # ─── MAIN LOGIC ────────────────────────────────────────
    if img0 is not None and run and orig_w and orig_h:
        # determine name for saving
        if uploaded:
            name = os.path.splitext(uploaded.name)[0] + ".jpg"
        else:
            name = os.path.splitext(sample_img)[0] + ".jpg"

        testd = os.path.join(args.data_dir, "data", "test")
        os.makedirs(testd, exist_ok=True)
        cv2.imwrite(os.path.join(testd, name), img0)

        orig_init = BaseDataset.__init__
        def patched_init(self, data_dir, phase, input_h=None, input_w=None, down_ratio=4):
            orig_init(self, data_dir, phase, input_h, input_w, down_ratio)
            if phase == "test":
                self.img_ids = [name]
        BaseDataset.__init__ = patched_init

        with st.spinner("Running model…"):
            net = Network(args)
            net.test(args, save=True)

        out_dir = os.path.join(REPO, f"results_{args.dataset}")
        pred_file = [f for f in os.listdir(out_dir)
                     if f.startswith(name) and f.endswith("_pred.jpg")][0]
        txtf = os.path.join(out_dir, f"{name}.txt")
        imgf = os.path.join(out_dir, pred_file)

        # ─── Annotated Predictions ─────────────────────────
        base = cv2.imread(imgf)
        txt = np.loadtxt(txtf)
        tlx, tly = txt[:, 2].astype(int), txt[:, 3].astype(int)
        trx, try_ = txt[:, 4].astype(int), txt[:, 5].astype(int)
        blx, bly = txt[:, 6].astype(int), txt[:, 7].astype(int)
        brx, bry = txt[:, 8].astype(int), txt[:, 9].astype(int)

        top_pts, bot_pts, mids, dists = [], [], [], []
        for (x1, y1), (x2, y2), (x3, y3), (x4, y4) in zip(
                zip(tlx, tly), zip(trx, try_),
                zip(blx, bly), zip(brx, bry)):
            tm = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            bm = np.array([(x3 + x4) / 2, (y3 + y4) / 2])
            top_pts.append(tm)
            bot_pts.append(bm)
            mids.append((tm + bm) / 2)
            dists.append(np.linalg.norm(bm - tm))

        ref = dists[-1]
        ann = base.copy()
        for tm, bm in zip(top_pts, bot_pts):
            cv2.line(ann, tuple(tm.astype(int)), tuple(bm.astype(int)), (0, 255, 255), 2)
        for m, d in zip(mids, dists):
            pct = (d - ref) / ref * 100
            clr = (0, 255, 255) if pct <= 20 else (0, 165, 255) if pct <= 40 else (0, 0, 255)
            pos = (int(m[0]) + 40, int(m[1]) + 5)
            cv2.putText(ann, f"{pct:.0f}%", pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, clr, 2, cv2.LINE_AA)

        ann_resized = cv2.resize(ann, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        with col5:
            st.image(cv2.cvtColor(ann_resized, cv2.COLOR_BGR2RGB), use_container_width=True)

        H, W = base.shape[:2]
        heat = np.zeros((H, W), np.float32)
        for cx, cy in [(int(m[0]), int(m[1])) for m in mids]:
            blob = np.zeros_like(heat)
            blob[cy, cx] = 1.0
            heat += cv2.GaussianBlur(blob, (0, 0), sigmaX=8, sigmaY=8)
        heat /= (heat.max() + 1e-8)
        hm8 = (heat * 255).astype(np.uint8)
        hm_c = cv2.applyColorMap(hm8, cv2.COLORMAP_JET)

        raw = cv2.imread(imgf, cv2.IMREAD_GRAYSCALE)
        raw_b = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(raw_b, 0.6, hm_c, 0.4, 0)
        overlay_resized = cv2.resize(overlay, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        with col6:
            st.image(cv2.cvtColor(overlay_resized, cv2.COLOR_BGR2RGB), use_container_width=True)

elif feature == "AP - Cobb angle":
    st.write("กำลังพัฒนา")

elif feature == "LA - Image Segmetation":
    uploaded = st.file_uploader("", type=["jpg", "jpeg", "png"])
    img0 = None

    # ─── Maintain selected sample in session state ─────────
    if "sample_img_la" not in st.session_state:
        st.session_state.sample_img_la = None

    # ─── SAMPLE BUTTONS ─────────────────────────────────────
    with col1:
        if st.button(" 1️⃣ Example ", use_container_width=True):
            st.session_state.sample_img_la = "image_1_la.jpg"
    with col2:
        if st.button(" 2️⃣ Example ", use_container_width=True):
            st.session_state.sample_img_la = "image_2_la.jpg"
    with col3:
        if st.button(" 3️⃣ Example ", use_container_width=True):
            st.session_state.sample_img_la = "image_3_la.jpg"

    # ─── UI FOR UPLOAD + DISPLAY ───────────────────────────
    run_la = st.button("Enter", use_container_width=True)
    col7, col8 = st.columns(2)

    with col7:
        st.subheader("🖼️ Original Image")

        sample_img_la = st.session_state.sample_img_la  # read persisted choice

        # case 1: uploaded file
        if uploaded:
            buf = uploaded.getvalue()
            img0 = Image.open(BytesIO(buf)).convert("RGB")
            st.image(img0, caption="Uploaded Image", use_container_width=True)

        # case 2: selected sample image
        elif sample_img_la is not None:
            img_path = os.path.join(REPO, sample_img_la)
            if os.path.isfile(img_path):
                img0 = Image.open(img_path).convert("RGB")
                st.image(img0, caption=f"Sample Image: {sample_img_la}", use_container_width=True)
            else:
                st.error(f"Cannot find {sample_img_la} in directory!")

    with col8:
        st.subheader("🔎 Predicted Image")

        # ─── PREDICTION ────────────────────────────────────
        if img0 is not None and run_la:
            img_np = np.array(img0)
            model = YOLO('./best.pt')  # or your correct path to best.pt
            with st.spinner("Running YOLO model…"):
                results = model(img_np, imgsz=640)
            pred_img = results[0].plot(boxes=False, probs=False)  # returns numpy image with annotations
            st.image(pred_img, caption="Prediction Result", use_container_width=True)

elif feature == "Contract":
    with col1:
        st.image("dev_1.jpg", caption=None, use_container_width=True)
        st.markdown(
            """
            <div style='border:2px solid #0080FF; border-radius:10px; padding:15px; text-align:center; background-color:#F0F8FF'>
                <h3>Thitsanapat Uma</h3>
                <a href='https://www.facebook.com/thitsanapat.uma' target='_blank'>
                    🔗 Facebook Profile
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.image("dev_2.jpg", caption=None, use_container_width=True)
        st.markdown(
            """
            <div style='border:2px solid #0080FF; border-radius:10px; padding:15px; text-align:center; background-color:#F0F8FF'>
                <h3>Santipab Tongchan</h3>
                <a href='https://www.facebook.com/santipab.tongchan.2025' target='_blank'>
                    🔗 Facebook Profile
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col3:
        st.image("dev_3.jpg", caption=None, use_container_width=True)
        st.markdown(
            """
            <div style='border:2px solid #0080FF; border-radius:10px; padding:15px; text-align:center; background-color:#F0F8FF'>
                <h3>Suphanat Kamphapan</h3>
                <a href='https://www.facebook.com/suphanat.kamphapan' target='_blank'>
                    🔗 Facebook Profile
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )



