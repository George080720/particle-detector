import streamlit as st
import cv2
import numpy as np
from scipy.spatial import distance_matrix

st.title("粒子检测工具")

# ========== 1. 上传图片 ==========
uploaded = st.file_uploader("上传 SEM 图", type=['png', 'jpg'])

if uploaded:
    # 从上传的文件字节流解码图片
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if img is None:
        st.error("图片读取失败，请换一张图。")
        st.stop()

    st.image(img, caption="原图", use_column_width=True)

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

    # 防止 lower > upper
    if l1 > u1:
        l1 = u1
    if l2 > u2:
        l2 = u2

    # ========== 3. 染色预览 ==========
    mask1 = cv2.inRange(img, l1, u1)   # 暗心
    mask2 = cv2.inRange(img, l2, u2)   # 亮圈
    mask = cv2.bitwise_or(mask1, mask2)

    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img_color[mask1 > 0] = [0, 0, 255]    # 红色（暗心）
    img_color[mask2 > 0] = [255, 0, 0]    # 蓝色（亮圈）

    st.image(img_color, caption="染色预览", use_column_width=True)

    # ========== 4. 检测 ==========
    if st.button("检测"):

        # 形态学去噪
        final_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

        # 连通域分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(final_mask)

        props = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            x, y = centroids[i]
            if area < 3:
                continue
            props.append((y, x, area))

        # 去重
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

        # ========== 5. 输出：染色图 + 底部结论文字 ==========
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

        # 显示结果
        st.image(out_img, caption="检测结果", use_column_width=True)
        st.success(f"粒子数：{len(props)}")
        st.info(f"平均间距：{avg_spacing:.1f} px")

        # 下载结果图
        result_path = "result.png"
        cv2.imwrite(result_path, out_img)
        with open(result_path, "rb") as f:
            st.download_button("下载结果图", f, file_name="result.png")
