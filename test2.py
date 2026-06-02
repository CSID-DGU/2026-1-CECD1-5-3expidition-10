import numpy as np
import cv2
import colorsys
import matplotlib.pyplot as plt

def get_line_intersection(line1, line2):
    """직선의 일반형 Ax + By = C 형태로 변환하여 안정적으로 교차점을 계산"""
    vx1, vy1, x1, y1 = line1.flatten()
    vx2, vy2, x2, y2 = line2.flatten()
    
    A1, B1, C1 = -vy1, vx1, vx1 * y1 - vy1 * x1
    A2, B2, C2 = -vy2, vx2, vx2 * y2 - vy2 * x2
    
    det = A1 * B2 - A2 * B1
    if abs(det) < 1e-5:
        return None
        
    x = (C1 * B2 - C2 * B1) / det
    y = (A1 * C2 - A2 * C1) / det
    return int(round(x)), int(round(y))

def get_robust_average_quadrilateral(pts):
    """회전 방어형 외각선 추출 및 네 꼭짓점 반환 함수"""
    points = pts.reshape(-1, 2).astype(np.float32)
    rect = cv2.minAreaRect(points.astype(np.int32))
    (cx, cy), _, angle = rect
    
    M_rot = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    ones = np.ones(shape=(len(points), 1), dtype=np.float32)
    points_homo = np.hstack([points, ones])
    rotated_points = M_rot.dot(points_homo.T).T

    top_pts, bottom_pts, left_pts, right_pts = [], [], [], []
    for orig_pt, rot_pt in zip(points, rotated_points):
        rx, ry = rot_pt[0], rot_pt[1]
        dx, dy = rx - cx, ry - cy
        rot_angle = np.arctan2(dy, dx) * 180 / np.pi
        
        if -45 <= rot_angle < 45:    right_pts.append(orig_pt)
        elif 45 <= rot_angle < 135:  bottom_pts.append(orig_pt)
        elif rot_angle >= 135 or rot_angle < -135: left_pts.append(orig_pt)
        else:                        top_pts.append(orig_pt)

    if min(len(top_pts), len(bottom_pts), len(left_pts), len(right_pts)) < 2:
        return np.int32(cv2.boxPoints(rect)), None # None을 추가하여 형식을 맞춤

    # 각 그룹별 Line Fitting
    line_t = cv2.fitLine(np.array(top_pts, dtype=np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01)
    line_b = cv2.fitLine(np.array(bottom_pts, dtype=np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01)
    line_l = cv2.fitLine(np.array(left_pts, dtype=np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01)
    line_r = cv2.fitLine(np.array(right_pts, dtype=np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01)

    pt_top_left = get_line_intersection(line_t, line_l)
    pt_top_right = get_line_intersection(line_t, line_r)
    pt_bottom_right = get_line_intersection(line_b, line_r)
    pt_bottom_left = get_line_intersection(line_b, line_l)
    
    quad_points = [pt_top_left, pt_top_right, pt_bottom_right, pt_bottom_left]
    
    if None in quad_points:
        return np.int32(cv2.boxPoints(rect)), None
        
    return np.array(quad_points, dtype=np.int32), (top_pts, bottom_pts, left_pts, right_pts)


# ==========================================
# 🧪 가상 데이터 생성 및 시각화 테스트 코드
# ==========================================
if __name__ == "__main__":
    # 1. 완벽한 사각형의 가상 꼭짓점 정의 (임의로 25도 회전된 사각형 유도)
    base_box = np.array([[150, 100], [450, 130], [400, 430], [100, 400]], dtype=np.float32)
    
    # 2. 각 변을 쭈글쭈글하고 울퉁불퉁한 다수의 점들로 채우기 (노이즈 추가)
    test_points = []
    for i in range(4):
        p1 = base_box[i]
        p2 = base_box[(i + 1) % 4]
        # 변 하나당 50개의 점 생성
        for t in np.linspace(0, 1, 50):
            pt = p1 * (1 - t) + p2 * t
            # 왜곡 및 노이즈 추가 (Huber Loss 테스트용 튀는 점 포함)
            noise_x = np.random.normal(0, 3) 
            noise_y = np.random.normal(0, 3)
            if np.random.rand() > 0.95:  # 5% 확률로 뜬금없이 크게 튀는 노이즈 점 생성
                noise_x += np.random.choice([-20, 20])
                noise_y += np.random.choice([-20, 20])
            test_points.append([pt[0] + noise_x, pt[1] + noise_y])
            
    pts_array = np.array(test_points, dtype=np.float32)

    # 수정: quad_pts만 가져오도록 튜플 언패킹
    quad_pts, _ = get_robust_average_quadrilateral(pts_array)

    plt.figure(figsize=(10, 8))
    plt.scatter(pts_array[:, 0], pts_array[:, 1], color='gray', alpha=0.6, label='Original Points')

    # 수정: quad_pts 사용
    polygon_pts = np.vstack([quad_pts, quad_pts[0]])
    plt.plot(polygon_pts[:, 0], polygon_pts[:, 1], color='red', linestyle='-', linewidth=2.5, label='Fitted Quadrilateral')
    plt.scatter(quad_pts[:, 0], quad_pts[:, 1], color='black', edgecolor='white', s=120, zorder=5, label='Corners')

    plt.gca().invert_yaxis()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.axis('equal')
    plt.show()