import json
import math
import sys
from collections import defaultdict

def get_connected_edges_assignments(vertex_index, edges_vertices, edges_assignment):
    """指定された頂点に接続するエッジの種類（山折り、谷折りなど）を取得する"""
    assignments = []
    for i, edge in enumerate(edges_vertices):
        if vertex_index in edge:
            assignments.append(edges_assignment[i])
    return assignments

def get_angles_around_vertex(vertex_index, vertices_coords, edges_vertices):
    """指定された頂点の周りの角度を計算してリストで返す"""
    center_point = vertices_coords[vertex_index]
    
    # 頂点に接続する他の頂点の座標を取得
    connected_points = []
    for edge in edges_vertices:
        if edge[0] == vertex_index:
            connected_points.append(vertices_coords[edge[1]])
        elif edge[1] == vertex_index:
            connected_points.append(vertices_coords[edge[0]])

    if len(connected_points) < 2:
        return []

    # 各点と中心点との角度を計算
    angles_points = []
    for p in connected_points:
        angle = math.atan2(p[1] - center_point[1], p[0] - center_point[0])
        angles_points.append((angle, p))
    
    # 角度でソート
    angles_points.sort()

    # ソートされた点の間（セクター）の角度を計算
    sector_angles = []
    num_points = len(angles_points)
    for i in range(num_points):
        angle1, _ = angles_points[i]
        angle2, _ = angles_points[(i + 1) % num_points]
        diff = angle2 - angle1
        # 角度が負になる場合（360度をまたぐ場合）は2πを足す
        if diff < 0:
            diff += 2 * math.pi
        sector_angles.append(diff)
        
    return sector_angles

def check_maekawa_theorem(assignments, vertex_index):
    """前川の定理を検証する: |M - V| = 2"""
    m_count = assignments.count("M")
    v_count = assignments.count("V")
    if abs(m_count - v_count) != 2:
        return {
            "type": "Maekawa",
            "vertex": vertex_index,
            "message": f"Vertex {vertex_index} fails Maekawa's theorem. |M({m_count}) - V({v_count})| = {abs(m_count - v_count)}, but should be 2."
        }
    return None

def check_kawasaki_theorem(angles, vertex_index):
    """川崎の定理を検証する: 交互の角度の和がそれぞれ180度になる"""
    if not angles or len(angles) % 2 != 0:
        # 奇数個の角度では定理は適用できない
        return {
            "type": "Kawasaki",
            "vertex": vertex_index,
            "message": f"Vertex {vertex_index} has an odd number of angles ({len(angles)}), which is not typical for flat-foldable interior vertices."
        }

    # 角度の総和が360度(2π)であるかチェック
    if not math.isclose(sum(angles), 2 * math.pi):
        return {
            "type": "Kawasaki",
            "vertex": vertex_index,
            "message": f"Vertex {vertex_index} angles do not sum to 360 degrees. Sum is {math.degrees(sum(angles)):.2f}."
        }
    
    # 交互の角度の和を計算
    odd_sum = sum(angles[i] for i in range(0, len(angles), 2))
    even_sum = sum(angles[i] for i in range(1, len(angles), 2))

    if not math.isclose(odd_sum, even_sum):
        return {
            "type": "Kawasaki",
            "vertex": vertex_index,
            "message": (
                f"Vertex {vertex_index} fails Kawasaki's theorem. "
                f"Alternating angle sums are not equal. "
                f"Odd sum: {math.degrees(odd_sum):.2f}, Even sum: {math.degrees(even_sum):.2f}."
            )
        }
    return None

def check_big_little_big_theorem(angles, vertex_index):
    """
    大小大の定理（Big-Little-Big Theorem）を検証する。
    折り線の数が4本の頂点にのみ適用される。
    最小角度と最大角度の和は、他の2つの角度の和以下でなければならない。
    (smallest + largest <= middle1 + middle2)
    """
    if len(angles) != 4:
        return None # この定理は折り線が4本の頂点にのみ適用

    sorted_angles = sorted(angles)
    smallest_plus_largest = sorted_angles[0] + sorted_angles[3]
    middle_sum = sorted_angles[1] + sorted_angles[2]

    # 浮動小数点数の比較のため、A > B を A - B > tolerance でチェック
    if smallest_plus_largest > middle_sum and not math.isclose(smallest_plus_largest, middle_sum):
         return {
            "type": "BigLittleBig",
            "vertex": vertex_index,
            "message": (
                f"Vertex {vertex_index} fails the Big-Little-Big theorem for degree 4 vertices. "
                f"Sum of smallest and largest angles ({math.degrees(smallest_plus_largest):.2f} deg) "
                f"must not be greater than the sum of the other two angles ({math.degrees(middle_sum):.2f} deg)."
            )
        }
    return None

def validate_fold_file(file_path):
    """
    .foldファイルを読み込み、平坦折り畳み可能性のルールを検証する
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        return {"valid": False, "errors": [{"type": "File Error", "message": str(e)}]}

    # 必須キーのチェック
    required_keys = ["vertices_coords", "edges_vertices", "edges_assignment"]
    for key in required_keys:
        if key not in data:
            return {"valid": False, "errors": [{"type": "Format Error", "message": f"Missing required key: '{key}'"}]}

    vertices_coords = data["vertices_coords"]
    edges_vertices = data["edges_vertices"]
    edges_assignment = data["edges_assignment"]
    
    errors = []

    # 各頂点についてループ
    for i in range(len(vertices_coords)):
        # 頂点に接続するエッジの種類を取得
        assignments = get_connected_edges_assignments(i, edges_vertices, edges_assignment)
        
        # 境界("B")に接続する頂点は内点ではないため、定理のチェックをスキップ
        if "B" in assignments:
            continue
        
        # --- レベル1検証 ---
        
        # 1. 前川の定理
        error = check_maekawa_theorem(assignments, i)
        if error:
            errors.append(error)

        # 頂点周りの角度を計算
        angles = get_angles_around_vertex(i, vertices_coords, edges_vertices)
        if not angles:
            continue
            
        # 2. 川崎の定理
        error = check_kawasaki_theorem(angles, i)
        if error:
            errors.append(error)
        
        # 3. 大小大の定理 (新規追加)
        error = check_big_little_big_theorem(angles, i)
        if error:
            errors.append(error)

    if errors:
        return {"valid": False, "errors": errors}
    
    return {"valid": True, "errors": []}

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python validator_v4.py <path_to_fold_file>")
        sys.exit(1)
        
    file_path = sys.argv[1]
    result = validate_fold_file(file_path)
    print(json.dumps(result, indent=2))
