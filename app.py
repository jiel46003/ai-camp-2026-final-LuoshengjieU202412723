"""NBA 场均得分预测器（Streamlit 网页）。

运行：
    streamlit run app.py

依赖：
    python -m pip install -r requirements.txt   （含 streamlit）

原理：
    读取 main.py 训练时保存的模型与标准化参数（models/model.joblib、models/scaler.json），
    对用户输入做与训练时完全相同的 z-score 标准化后预测。
    若模型文件不存在，先运行 `python main.py` 生成它们。

边界：
    只提供"粗略得分期望"；对伤病、体系与技术流球员可能严重偏差，结果需人工复核。
"""

import json
from pathlib import Path

import joblib
import streamlit as st

MODEL_PATH = Path("models/model.joblib")
SCALER_PATH = Path("models/scaler.json")

# 与 main.py 的 NUMERIC_FEATURES 顺序保持一致
# 训练时特征 = [1.0, age, player_height, player_weight, draft_round, draft_number]
FEATURE_ORDER = ["age", "player_height", "player_weight", "draft_round", "draft_number"]


@st.cache_resource
def load_model():
    """加载模型与标准化参数（Streamlit 会缓存，避免每次交互重新读文件）。"""
    model = joblib.load(MODEL_PATH)
    scaler = json.loads(SCALER_PATH.read_text(encoding="utf-8"))
    return model, scaler


def predict_score(model, scaler, values):
    """把用户输入变成训练时的特征并预测。values 顺序必须与 FEATURE_ORDER 一致。"""
    means, stds = scaler["means"], scaler["stds"]
    row = [1.0]  # 常数项，与 build_xy 一致
    for c, v in zip(FEATURE_ORDER, values):
        row.append((v - means[c]) / stds[c])
    return float(model.predict([row])[0])


st.set_page_config(page_title="NBA 场均得分预测器", page_icon="🏀")
st.title("🏀 NBA 场均得分预测器")
st.caption("根据进入联盟前可知的身体与选秀信息，快速得到一个场均得分期望。")

if not (MODEL_PATH.exists() and SCALER_PATH.exists()):
    st.error("未找到模型文件。请先运行 `python main.py` 生成 "
             "models/model.joblib 和 models/scaler.json。")
    st.stop()

model, scaler = load_model()

with st.form("player_form"):
    st.subheader("输入球员信息（选秀前可知）")
    age = st.number_input("年龄（岁）", min_value=17, max_value=45, value=22, step=1)
    height = st.number_input("身高（cm）", min_value=150, max_value=240, value=201, step=1)
    weight = st.number_input("体重（kg）", min_value=60, max_value=180, value=104, step=1)
    draft_round = st.selectbox("选秀轮次（落选 = 0）", [0, 1, 2])
    draft_number = st.number_input("选秀顺位（落选 = 0）", min_value=0, max_value=60, value=30, step=1)
    submitted = st.form_submit_button("预测得分")

if submitted:
    pred = predict_score(
        model, scaler,
        [age, height, weight, float(draft_round), float(draft_number)],
    )
    st.metric("预测场均得分", f"{pred:.1f} 分/场")
    st.info("提示：这是基于静态信息的粗略期望。伤病、体系与技术风格无法由这些特征解释，"
            "请人工复核后再做决定。")
