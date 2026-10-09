import streamlit as st
import cv2
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix

st.title("粒子检测工具")

# ========== 1. 上传图片 ==========
uploaded = st.file_uploader("上传 SEM 图", type=['png', 'jpg'])

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if img is None:
        st.error("图片读取失败，请换一张图。")
        st.stop()

    st.image(img, caption="原图", use_container_width=True)

    # ========== 2. 滑动条 ==========
    st.subheader("染色区间")
    col1, col2 = st.columns(2)
    with col1:
        st.write("暗心区间")
        l1 = st.slider("L1", 0, 255, 40)
        u1 = st.slider("U1", 0, 255, 80)
    with col2:
        st.write("亮圈区间")
        l2 = st.slider("L2", 0, 255, 154)
        u2 = st.slider("U2", 0, 255, 245)

    if l1 > u1:
        l1 = u1
    if l2 > u2:
        l2 = u2

    # ========== 3. 染色预览 ==========
    mask1 = cv2.inRange(img, l1, u1)
    mask2 = cv2.inRange(img, l2, u2)
    mask = cv2.bitwise_or(mask1, mask2)

    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img_color[mask1 > 0] = [0, 0, 255]
    img_color[mask2 > 0] = [255, 0, 0]

    st.image(img_color, caption="染色预览", use_container_width=True)

    # ========== 4. 检测 ==========
    if st.button("检测"):

        final_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(final_mask)

        props = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            x, y = centroids[i]
            if area < 3:
                continue
            props.append((y, x, area))

        def merge_close(points, min_dist=5):
            merged = []
            for y, x, area in points:
                too_close = False
                for my, mx, ma in merged:
                    if np.sqrt((y - my)**2 + (x - mx)**2) < min_dist:
                        too_close = True
                        break
                if not too_close:
                    merged.append((y, x, area))
            return merged

        props = merge_close(props, min_dist=5)

        # 计算间距
        if len(props) >= 2:
            coords = np.array([(p[0], p[1]) for p in props])
            dist_mat = distance_matrix(coords, coords)
            np.fill_diagonal(dist_mat, np.inf)
            nearest_dist = dist_mat.min(axis=1)
            avg_spacing = nearest_dist.mean()
        else:
            avg_spacing = 0
            nearest_dist = np.array([])

        # ========== 5. 输出：染色图 + 结论文字 ==========
        out_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        out_img[mask1 > 0] = [0, 0, 255]
        out_img[mask2 > 0] = [255, 0, 0]

        h, w = out_img.shape[:2]
        text1 = f"Particles: {len(props)}"
        text2 = f"Avg Spacing: {avg_spacing:.1f} px"
        cv2.putText(out_img, text1, (10, h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 4)
        cv2.putText(out_img, text1, (10, h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(out_img, text2, (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 4)
        cv2.putText(out_img, text2, (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        st.image(out_img, caption="检测结果", use_container_width=True)
        st.success(f"粒子数：{len(props)}")
        st.info(f"平均间距：{avg_spacing:.1f} px")

        # ========== 6. 导出 CSV ==========
        df = pd.DataFrame({
            'x': [p[1] for p in props],
            'y': [p[0] for p in props],
            'area': [p[2] for p in props],
            'nearest_dist': nearest_dist if len(nearest_dist) > 0 else [0]*len(props)
        })
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("下载 CSV", csv, file_name="particles.csv")

        # ========== 7. 人工验证 ==========
        st.subheader("人工验证")
        st.write("随机抽取 10 个粒子，判断它们是否是真实的粒子。")

        np.random.seed(42)
        sample_size = min(10, len(props))
        sample_idx = np.random.choice(len(props), sample_size, replace=False)

        if 'verify_results' not in st.session_state:
            st.session_state.verify_results = {}

        for idx in sample_idx:
            y, x, area = props[idx]

            # ★ 在原图上标出选中的粒子（绿色圆圈）
            img_marked = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            radius = int(np.sqrt(area / np.pi))
            cv2.circle(img_marked, (int(x), int(y)), max(radius, 2), (0, 255, 0), 1)

            # 截取局部区域
            y0, y1 = max(0, int(y)-30), min(img.shape[0], int(y)+30)
            x0, x1 = max(0, int(x)-30), min(img.shape[1], int(x)+30)
            patch = img_marked[y0:y1, x0:x1]

            # 放大 5 倍
            patch_big = cv2.resize(patch, None, fx=5, fy=5, interpolation=cv2.INTER_NEAREST)

            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                st.image(patch_big, caption=f"粒子 {idx}", width=150)
            with col2:
                st.write(f"面积：{area}")
            with col3:
                choice = st.radio(f"判断 {idx}", ["对", "错"], key=f"verify_{idx}")
                st.session_state.verify_results[idx] = choice

        if st.button("计算误检率"):
            results = st.session_state.verify_results
            total = len(results)
            correct = sum(1 for v in results.values() if v == "对")
            wrong = total - correct
            error_rate = wrong / total * 100 if total > 0 else 0

            st.success(f"验证结果：{correct}/{total} 正确，误检率 {error_rate:.1f}%")
