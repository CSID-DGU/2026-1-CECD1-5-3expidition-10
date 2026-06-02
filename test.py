import numpy as np
import cv2
import matplotlib.pyplot as plt

def get_line_intersection(line1, line2):
    """두 직선의 일반형 Ax + By = C 교차점 계산"""
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
    points = pts.reshape(-1, 2).astype(np.float32)
    
    # 1. 기준이 될 최소 사각형(박스) 꼭짓점 4개 추출
    rect = cv2.minAreaRect(points.astype(np.int32))
    box_pts = cv2.boxPoints(rect)  # [P0, P1, P2, P3] 회전된 순서대로 4개 점
    
    # 4개의 변(Line) 정의하기
    # 각 변을 (vx, vy, x0, y0) 형태로 빌드
    lines = []
    for i in range(4):
        p1 = box_pts[i]
        p2 = box_pts[(i + 1) % 4]
        vec = p2 - p1
        mag = np.linalg.norm(vec)
        vx, vy = vec / (mag + 1e-6)
        lines.append((vx, vy, p1[0], p1[1]))
        
    # 2. 모든 점들을 가장 '거리가 가까운 변' 그룹으로 분류 (거리 공식 활용)
    groups = [[], [], [], []]
    for pt in points:
        px, py = pt[0], pt[1]
        distances = []
        for vx, vy, x0, y0 in lines:
            # 점과 직선 사이의 거리 공식
            dist = abs(-vy * px + vx * py - (vx * y0 - vy * x0))
            distances.append(dist)
        
        best_edge_idx = np.argmin(distances)
        groups[best_edge_idx].append(pt)
        
    # 예외 처리: 한 변에 점이 너무 없으면 팅구기
    if min(len(g) for g in groups) < 2:
        return np.int32(box_pts), groups

    # 3. ★핵심★ DIST_HUBER를 이용해 각 변의 진짜 평균 직선 피팅
    fitted_lines = []
    for g in groups:
        f_line = cv2.fitLine(np.array(g, dtype=np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01)
        fitted_lines.append(f_line)

    # 4. 피팅된 직선들의 교차점을 순서대로 연결하여 꼭짓점 추출
    quad_points = []
    for i in range(4):
        pt = get_line_intersection(fitted_lines[i], fitted_lines[(i - 1) % 4])
        quad_points.append(pt)
        
    if None in quad_points:
        return np.int32(box_pts), groups
        
    return np.array(quad_points, dtype=np.int32), groups

# ==========================================
# 🧪 테스트 실행 코드
# ==========================================
if __name__ == "__main__":
    # 일부러 가로세로 비율이 다르고 노이즈가 심한 사각형 점 생성
    base_box = np.array([[120, 100], [480, 140], [420, 420], [80, 380]], dtype=np.float32)
    
    test_points = []
    for i in range(4):
        p1 = base_box[i]
        p2 = base_box[(i + 1) % 4]
        for t in np.linspace(0, 1, 60):
            pt = p1 * (1 - t) + p2 * t
            # 가우시안 노이즈 추가
            noise_x = np.random.normal(0, 4)
            noise_y = np.random.normal(0, 4)
            # 아웃라이어(튀는 점) 무작위 유도
            if np.random.rand() > 0.94:
                noise_x += np.random.choice([-35, 35])
                noise_y += np.random.choice([-35, 35])
            test_points.append([pt[0] + noise_x, pt[1] + noise_y])
            
    pts_array = np.array(test_points, dtype=np.float32)

    # 로직 수행
    quad_results, groups = get_robust_average_quadrilateral(pts_array)

    # 시각화
    plt.figure(figsize=(10, 8))
    
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple']
    for idx, g in enumerate(groups):
        g = np.array(g)
        if len(g) > 0:
            plt.scatter(g[:, 0], g[:, 1], color=colors[idx], alpha=0.6, label=f'Edge {idx} Pts')

    # 다각형 닫기
    polygon_pts = np.vstack([quad_results, quad_results[0]])
    plt.plot(polygon_pts[:, 0], polygon_pts[:, 1], color='red', linestyle='-', linewidth=2.5, label='True Huber Average')
    plt.scatter(quad_results[:, 0], quad_results[:, 1], color='black', edgecolor='white', s=120, zorder=5)

    plt.title("Fixed Distance-based Robust Fitting", fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.axis('equal')
    plt.show()