import cv2
import numpy as np
from scipy.spatial import distance_matrix

# ========== 1. 读图 ==========
filename = r'C:\Users\User\Desktop\crystal_demo\Pre_run.png'
img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

if img is None:
    print(f"错误：找不到文件 {filename}")
    exit()

# ========== 2. 窗口和滑动条 ==========
window_name = "Dye Tool - 's' detect, 'q' quit"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# 区间1（暗心）
cv2.createTrackbar("L1", window_name, 40, 255, lambda x: None)
cv2.createTrackbar("U1", window_name, 80, 255, lambda x: None)
# 区间2（亮圈）
cv2.createTrackbar("L2", window_name, 154, 255, lambda x: None)
cv2.createTrackbar("U2", window_name, 245, 255, lambda x: None)

# ========== 3. 循环：实时更新染色 ==========
while True:
    l1 = cv2.getTrackbarPos("L1", window_name)
    u1 = cv2.getTrackbarPos("U1", window_name)
    l2 = cv2.getTrackbarPos("L2", window_name)
    u2 = cv2.getTrackbarPos("U2", window_name)

    if l1 > u1:
        l1 = u1
    if l2 > u2:
        l2 = u2

    # 两个区间的掩膜
    mask1 = cv2.inRange(img, l1, u1)   # 暗心
    mask2 = cv2.inRange(img, l2, u2)   # 亮圈
    mask = cv2.bitwise_or(mask1, mask2)  # 合并

    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img_color[mask1 > 0] = [0, 0, 255]    # 红色（暗心）
    img_color[mask2 > 0] = [255, 0, 0]    # 蓝色（亮圈）

    count = np.sum(mask > 0)
    text = f"Dark: {l1}-{u1} | Bright: {l2}-{u2} | Pixels: {count}"
    cv2.putText(img_color, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    cv2.putText(img_color, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow(window_name, img_color)
    key = cv2.waitKey(50) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('s'):
        # ★ 检测
        final_mask = mask.copy()
        break

cv2.destroyAllWindows()

# ========== 4. 检测 ==========
print(f"染色掩膜像素数：{np.sum(final_mask > 0)}")

# 形态学去噪
final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

# 连通域分析
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(final_mask)

props = []
for i in range(1, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]
    x, y = centroids[i]
    if area < 3:
        continue
    props.append((y, x, area))

print(f"检测到 {len(props)} 个粒子")

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
print(f"去重后：{len(props)} 个粒子")

# 计算间距
if len(props) >= 2:
    coords = np.array([(p[0], p[1]) for p in props])
    dist_mat = distance_matrix(coords, coords)
    np.fill_diagonal(dist_mat, np.inf)
    nearest_dist = dist_mat.min(axis=1)
    avg_spacing = nearest_dist.mean()
    print(f"平均间距：{avg_spacing:.2f} 像素")
else:
    avg_spacing = 0

# ========== 5. 输出：染色图 + 底部结论文字 ==========
img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
img_color[mask1 > 0] = [0, 0, 255]
img_color[mask2 > 0] = [255, 0, 0]

h, w = img_color.shape[:2]
text1 = f"Particles: {len(props)}"
text2 = f"Avg Spacing: {avg_spacing:.1f} px"

cv2.putText(img_color, text1, (10, h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 4)
cv2.putText(img_color, text1, (10, h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
cv2.putText(img_color, text2, (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 4)
cv2.putText(img_color, text2, (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

output = filename.replace('.png', '_result.png')
cv2.imwrite(output, img_color)
print(f"已保存为 {output}")
