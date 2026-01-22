"""
눈 감지 및 맵 생성 모듈
카메라/센서 데이터에서 눈이 쌓인 영역을 찾아내고, AutoNavSim2D가 이해할 수 있는 맵 형식으로 변환
"""


class SnowDetector:
    """눈 영역 감지 클래스"""
    
    def __init__(self):
        """초기화 - 필요한 모델 로드 등"""
        pass
    
    def detect_snow(self, image):
        """
        이미지에서 눈이 쌓인 영역 감지
        
        Args:
            image: 입력 이미지 (numpy array 등)
        
        Returns:
            snow_mask: 눈 영역 마스크 (numpy array, 0=눈없음, 1=눈있음)
        
        TODO: 인지 팀원이 구현
        """
        raise NotImplementedError("인지 팀원이 구현해야 합니다")
    
    def generate_map(self, snow_mask):
        """
        눈 마스크를 AutoNavSim2D 맵 형식으로 변환
        
        Args:
            snow_mask: 눈 영역 마스크
        
        Returns:
            map_matrix: 2D 리스트 (1=이동가능, 0=장애물)
        
        """
        raise NotImplementedError("인지 팀원이 구현해야 합니다")





# 다중 군집화
# autonavsim2d에서 free space로 인식하는 색상은 WHITE, BLACK, GREEN, BLUE, RED, GREY, ORANGE 6가지입니다.
# 여기서는 RED가 목적지이고 WHITE는 쓸 수 없어 군집화 중 최대 4개만 free space로 인식됩니다.
# 맵으로 받은 코트장 좌표를 2D-List로 받고 이 Lidar정보를 다시 군집화 코드에 넣게끔 수정해야한다.
# 전체 눈 좌표, 군집화 목적지(여기선 중앙값), 좌표들 출력 등 함수들 판단에 넘겨주기 좋게 독립적으로 정리해야한다.

#아래 코드는 맵을 생성하기 위한 코드이다. autonavsim2d Map Generation부분 출처: https://github.com/yendiDev/autonavsim2d

# from autonavsim2d.autonavsim2d import AutoNavSim2D
#
# parameter configuration
# config = {
#     "show_frame": True,
#     "show_grid": False,
#     "map": None
# }
#
# nav = AutoNavSim2D(
#     custom_planner='default',
#     custom_motion_planner='default',
#     window='map_gen',
#     config=config
# )
#
# nav.run()

#snow_removal_area_multi3.pkl 맵을 생성했다. FILENAME에 넣어준다.








import pickle
import numpy as np
import random
from sklearn.cluster import DBSCAN
from autonavsim2d.autonavsim2d import AutoNavSim2D
from autonavsim2d.utils.utils import BLACK, GREEN, BLUE, GREY, ORANGE, RED

FILENAME = 'snow_removal_area_multi3.pkl'  #맵을 넣어준다.


# [1] 데이터 로드 로직
class FakeRect:
    def __init__(self, x, y, w, h): self.x, self.y = x, y


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == '__rect_constructor': return FakeRect
        return super().find_class(module, name)


def get_snow_area_list(file_path):
    with open(file_path, 'rb') as f:
        grid_data = CustomUnpickler(f).load()
    obstacle_list = []
    for row in grid_data:
        for cell in row:
            if cell[1] in [(0, 0, 0), [0, 0, 0], "black"]:
                obstacle_list.append(list(cell[2]))
    return obstacle_list


def get_unique_color(label):
    if label == -1: return (150, 150, 150)
    fixed_colors = [BLACK, GREEN, BLUE, GREY, ORANGE]
    if 0 <= label < len(fixed_colors): return fixed_colors[label]
    random.seed(int(label))
    return (random.randint(0, 50), random.randint(100, 255), random.randint(100, 255))


# 데이터 정리 핵심 함수
def get_perception_info(raw_points_list, labels, raw_points):
    perception_data = {
        "original_points": raw_points_list,
        "outliers": [],
        "clusters": []
    }
    unique_labels = set(labels)
    for label in sorted(unique_labels):
        indices = np.where(labels == label)[0]
        pts = raw_points[indices].tolist()
        if label == -1:
            perception_data["outliers"] = pts
            continue
        center = raw_points[indices].mean(axis=0)
        perception_data["clusters"].append({
            "cluster_id": int(label),
            "pixels": pts,
            "target_center": [round(center[0]), round(center[1])]
        })
    return perception_data


# --- [수정된 출력 함수] ---
def print_perception_info(raw_points_list, labels, raw_points):
    res = get_perception_info(raw_points_list, labels, raw_points)

    print("\n" + "=" * 30 + " DATA REPORT " + "=" * 30)
    print(f"📂 파일명: {FILENAME}")

    # 1. 모든 좌표 출력
    print(f"\n[1] 모든 눈 좌표 (총 {len(res['original_points'])}개):")
    print(res['original_points'])

    # 2. 이상치 출력
    print(f"\n[2] 이상치(Outliers) 좌표 (총 {len(res['outliers'])}개):")
    print(res['outliers'] if res['outliers'] else "이상치 없음")

    # 3. 군집별 정보 출력
    print(f"\n[3] 군집화 결과 (총 {len(res['clusters'])}개 군집):")
    for c in res['clusters']:
        print(f"  ▶ Cluster ID: {c['cluster_id']}")
        print(f"    - 중앙값(목적지): {c['target_center']}")
        print(f"    - 좌표 ({len(c['pixels'])}개): {c['pixels']}")
        print("-" * 40)
    print("=" * 73)


def simulate_perception_info(raw_points_list, labels, raw_points):
    res = get_perception_info(raw_points_list, labels, raw_points)
    config = {"show_frame": True, "show_grid": False, "map": FILENAME}
    nav = AutoNavSim2D(custom_planner='default', custom_motion_planner='default', window='amr', config=config)

    for c in res['clusters']:
        color = get_unique_color(c['cluster_id'])
        for r, c_idx in c['pixels']:
            nav.map_val[r][c_idx][1] = color
        tr, tc = c['target_center']
        nav.map_val[tr][tc][1] = RED

    print("\n시뮬레이터를 실행합니다...")
    nav.run()


if __name__ == "__main__":
    raw_points_list = get_snow_area_list(FILENAME)
    if raw_points_list:
        raw_points = np.array(raw_points_list)
        dbscan = DBSCAN(eps=10, min_samples=5).fit(raw_points)

        # 콘솔 출력 먼저 수행
        print_perception_info(raw_points_list, dbscan.labels_, raw_points)
        # 시뮬레이터 실행
        simulate_perception_info(raw_points_list, dbscan.labels_, raw_points)
    else:
        print("데이터를 불러오지 못했습니다.")
