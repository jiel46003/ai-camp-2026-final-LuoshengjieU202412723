"""最终项目（方案 A）：用身体与位置信息预测 NBA 球员场均得分（表格回归）。

一条命令运行：
    python main.py

依赖：Python 标准库 + scikit-learn（随机森林候选）。
      安装：python -m pip install -r requirements.txt
模型：基线 = 训练集平均得分；候选 = 随机森林回归（比基线复杂一步，可读特征重要性）。
流程：数据检查 → 数据清理 → 特征准备 → 固定划分 → 基线 → 候选 → 比较 MAE → 失败案例 → 保存结果。

学生应能逐行解释本文件；本文件不含虚构数据，真实数据需先放入 data/raw/。
"""

import csv
import random
import sys
from pathlib import Path

try:
    from sklearn.ensemble import RandomForestRegressor
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False

# ---- 数据契约（可配置）------------------------------------------------------
DATA_PATH = Path("data/raw/all_seasons.csv")          # 数据集文件（放 data/raw/，不提交 Git）
OUT_DIR = Path("results")                             # 结果输出目录

NUMERIC_FEATURES = [
    "age", "player_height", "player_weight",  # 身体信息（选秀前可知）
    "draft_round", "draft_number",            # 选秀信息（落选球员由 clean_data 编码为 0）
]
CATEGORIC_FEATURES = []  # 本数据集无 position 列；可选扩展：加 "country"
TARGET = "pts"           # 目标：场均得分

RANDOM_SEED = 42
TEST_FRACTION = 0.2        # 固定 20% 作为测试集


def mae(y_true, y_pred):
    """平均绝对误差：|真实 - 预测| 的平均值。"""
    return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / len(y_true)


# ---- 数据读取、清理与检查 -----------------------------------------------------
def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def clean_data(rows):
    """数据清理：把 draft_round/draft_number 中的非数值（如 'Undrafted'）编码为 0（落选=0）。"""
    n_replaced = 0
    for r in rows:
        for k in ("draft_round", "draft_number"):
            v = r.get(k)
            if v not in (None, ""):
                try:
                    float(v)
                except ValueError:
                    r[k] = "0"
                    n_replaced += 1
    if n_replaced:
        print(f"[clean] 将 {n_replaced} 个非数值选秀值编码为 0（落选=0）")
    return rows


def check_data(rows, header):
    """数据契约检查：只验证"拿到预期格式"，不证明内容全部正确。"""
    print(f"[check] 行数: {len(rows)}")
    print(f"[check] 实际列名: {header}")
    required = [TARGET] + NUMERIC_FEATURES
    missing = [c for c in required if c not in header]
    if missing:
        print(f"[check] 错误：缺少必需字段 {missing}")
        print("[check] 请核对字段名后修改 main.py 顶部 NUMERIC_FEATURES / TARGET。")
        return False
    for c in NUMERIC_FEATURES + [TARGET]:
        bad = sum(1 for r in rows if r.get(c) in (None, ""))
        if bad:
            print(f"[check] 字段 {c} 有 {bad} 行缺失/空值")
    print("[check] 数据契约检查通过（格式层面）。")
    return True


# ---- 特征与目标提取 -----------------------------------------------------------
def build_xy(rows, header):
    """返回 X（每行 = [1, 标准化数值..., one-hot 类别...]）、y（目标）、以及每行来源。"""
    pos_col = next((c for c in CATEGORIC_FEATURES if c in header), None)
    positions = sorted({r[pos_col] for r in rows if r.get(pos_col)}) if pos_col else []

    # z-score 标准化（用全量数据估计均值/标准差；课程里可改为只用训练集）
    means, stds = {}, {}
    for c in NUMERIC_FEATURES:
        vals = [float(r[c]) for r in rows if r.get(c) not in (None, "")]
        means[c] = sum(vals) / len(vals)
        stds[c] = (sum((v - means[c]) ** 2 for v in vals) / len(vals)) ** 0.5 or 1.0

    X, y, src = [], [], []
    for r in rows:
        if any(r.get(c) in (None, "") for c in NUMERIC_FEATURES + [TARGET]):
            continue  # 跳过缺失必需字段的行
        row = [1.0]  # 常数项
        for c in NUMERIC_FEATURES:
            row.append((float(r[c]) - means[c]) / stds[c])
        if pos_col:
            row += [1.0 if r.get(pos_col) == p else 0.0 for p in positions]
        X.append(row)
        y.append(float(r[TARGET]))
        src.append(r)
    return X, y, src


# ---- 主流程 -------------------------------------------------------------------
def main() -> int:
    if not DATA_PATH.exists():
        print("未找到数据文件：", DATA_PATH)
        print("请先下载数据集并放到 data/raw/（见 README 的『数据契约』一节）。")
        print("下载失败时保留错误信息，联系教师获取同一来源的缓存副本。")
        return 1

    rows = read_csv(DATA_PATH)
    header = list(rows[0].keys()) if rows else []
    if not rows:
        print("数据文件为空。")
        return 1
    if not check_data(rows, header):
        return 1

    rows = clean_data(rows)

    X, y, src = build_xy(rows, header)
    print(f"[data] 可用于训练/测试的样本数: {len(X)}")

    # 固定划分（随机 80/20），保证基线 vs 候选在同一测试集上比较
    rng = random.Random(RANDOM_SEED)
    idx = list(range(len(X)))
    rng.shuffle(idx)
    n_test = int(len(idx) * TEST_FRACTION)
    test_idx, train_idx = set(idx[:n_test]), idx[n_test:]
    X_tr = [X[i] for i in train_idx]; y_tr = [y[i] for i in train_idx]
    X_te = [X[i] for i in test_idx];  y_te = [y[i] for i in test_idx]

    # 基线：用训练集平均得分预测所有测试样本
    baseline = sum(y_tr) / len(y_tr)
    mae_base = mae(y_te, [baseline] * len(y_te))

    # 候选：随机森林（比基线复杂一步；可读特征重要性）
    if not HAVE_SKLEARN:
        print("\n未安装 scikit-learn，无法训练候选随机森林。请先运行：")
        print("    python -m pip install scikit-learn")
        return 1
    model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED)
    model.fit(X_tr, y_tr)
    y_pred = list(model.predict(X_te))
    mae_cand = mae(y_te, y_pred)

    print("\n===== 基线 vs 候选（同一测试集，%d 个样本）=====" % len(y_te))
    print(f"基线（平均分 {baseline:.2f}）        MAE = {mae_base:.3f} 分/场")
    print(f"候选（随机森林）            MAE = {mae_cand:.3f} 分/场")
    if mae_cand < mae_base:
        print(f"候选误差降低 {mae_base - mae_cand:.3f} 分/场（{(mae_base - mae_cand) / mae_base * 100:.1f}%）")
    else:
        print("候选未优于基线——记录这个诚实结论并分析原因。")

    # 特征重要性（可解释性：哪个特征对预测贡献最大）
    feat_names = ["常数项"] + list(NUMERIC_FEATURES)  # 若启用类别特征，需在此扩展对应列名
    print("\n===== 特征重要性（随机森林）=====")
    for name, imp in sorted(zip(feat_names, model.feature_importances_), key=lambda t: -t[1]):
        print(f"  {name}: {imp:.3f}")

    # 失败案例：测试集中误差最大的 3 个
    errors = sorted(
        zip(test_idx, y_te, y_pred),
        key=lambda t: abs(t[1] - t[2]),
        reverse=True,
    )[:3]
    print("\n===== 失败案例（测试集误差最大）=====")
    name_col = "player_name" if "player_name" in header else header[0]
    for i, yt, yp in errors:
        name = src[i].get(name_col, f"row{i}")
        print(f"{name}: 真实 {yt:.1f} / 预测 {yp:.1f} / 误差 {abs(yt - yp):.1f}")

    # 保存结果（结果文件可提交，原始数据不提交）
    OUT_DIR.mkdir(exist_ok=True)
    with open(OUT_DIR / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"测试样本数: {len(y_te)}\n")
        f.write(f"训练样本数: {len(y_tr)}\n")
        f.write(f"基线 MAE: {mae_base:.3f}\n")
        f.write(f"候选模型: 随机森林(n_estimators=100)\n")
        f.write(f"候选 MAE: {mae_cand:.3f}\n")
        f.write(f"候选改善: {mae_base - mae_cand:.3f} 分/场\n")
    print(f"\n结果已保存到 {OUT_DIR / 'summary.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
