import json
import math
import sys
from collections import defaultdict, namedtuple
import itertools

MAX_ERRORS_TO_REPORT = 10
EPSILON = 1e-9


def _get_orientation(p, q, r):
    """3点の位置関係（共線、時計回り、反時計回り）を返す"""
    # 浮動小数点数の計算誤差を考慮
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if math.isclose(val, 0, rel_tol=EPSILON, abs_tol=EPSILON):
        return 0  # Collinear
    return 1 if val > 0 else 2  # Clockwise or Counterclockwise


def _on_segment(p, q, r):
    """点qが線分pr上にあるかどうかをチェック（3点が共線であること前提）"""
    return (q[0] <= max(p[0], r[0]) + EPSILON and q[0] >= min(p[0], r[0]) - EPSILON and
            q[1] <= max(p[1], r[1]) + EPSILON and q[1] >= min(p[1], r[1]) - EPSILON)

def check_improper_edge_intersections(vertices_coords, edges_vertices):
    """
    エッジ同士が端点以外で交差、または不適切に重なっているかを検証する。(堅牢版)
    """
    errors = []
    edge_pairs = itertools.combinations(enumerate(edges_vertices), 2)

    for (idx1, edge1), (idx2, edge2) in edge_pairs:
        v1_idx, v2_idx = edge1
        v3_idx, v4_idx = edge2

        # 頂点を共有しているエッジはチェック対象外
        if len(set(edge1) & set(edge2)) > 0:
            continue

        p1, q1 = vertices_coords[v1_idx], vertices_coords[v2_idx]
        p2, q2 = vertices_coords[v3_idx], vertices_coords[v4_idx]

        o1 = _get_orientation(p1, q1, p2)
        o2 = _get_orientation(p1, q1, q2)
        o3 = _get_orientation(p2, q2, p1)
        o4 = _get_orientation(p2, q2, q1)

        # 一般的な交差ケース: 2つの線分が互いをまたいでいる
        if o1 != o2 and o3 != o4:
            errors.append({
                "type": "ImproperEdgeIntersection",
                "message": f"Edge {idx1} ({v1_idx}-{v2_idx}) and Edge {idx2} ({v3_idx}-{v4_idx}) improperly intersect.",
                "context": {"edge_indices": [idx1, idx2], "vertex_indices": [edge1, edge2]}
            })
            continue

        # 特殊ケース: 同一直線上にある場合の「重なり」をチェック
        # 重複チェックを避けるため、一度だけ判定する
        collinear_overlap_detected = False
        if o1 == 0 and _on_segment(p1, p2, q1):
            collinear_overlap_detected = True
        elif o2 == 0 and _on_segment(p1, q2, q1):
            collinear_overlap_detected = True
        elif o3 == 0 and _on_segment(p2, p1, q2):
            collinear_overlap_detected = True
        elif o4 == 0 and _on_segment(p2, q1, q2):
            collinear_overlap_detected = True

        if collinear_overlap_detected:
            errors.append({
                "type": "ImproperCollinearOverlap",
                "message": f"Edge {idx1} and Edge {idx2} are collinear and overlap.",
                "context": {"edge_indices": [idx1, idx2], "vertex_indices": [edge1, edge2]}
            })

    return errors if errors else None


def check_boundary_edges_on_frame(vertices_coords, edges_vertices, edges_assignment, x_range=(-200, 200), y_range=(-200, 200)):
    """
    境界線("B")が、指定された矩形領域の枠線と完全に一致する
    水平線または垂直線であることを検証する（シナリオB: 厳格ルール）。
    また、境界線の端点も枠上にあることを確認する。
    """
    errors = []
    x_min, x_max = min(x_range), max(x_range)
    y_min, y_max = min(y_range), max(y_range)

    for i, assignment in enumerate(edges_assignment):
        if assignment == "B":
            v1_idx, v2_idx = edges_vertices[i]
            x1, y1 = vertices_coords[v1_idx]
            x2, y2 = vertices_coords[v2_idx]

            # 端点が枠の境界上にあるかを事前にチェック
            def is_on_frame_boundary(x, y):
                on_left_right = (math.isclose(x, x_min, rel_tol=EPSILON, abs_tol=EPSILON) or 
                               math.isclose(x, x_max, rel_tol=EPSILON, abs_tol=EPSILON)) and (y_min - EPSILON <= y <= y_max + EPSILON)
                on_top_bottom = (math.isclose(y, y_min, rel_tol=EPSILON, abs_tol=EPSILON) or 
                               math.isclose(y, y_max, rel_tol=EPSILON, abs_tol=EPSILON)) and (x_min - EPSILON <= x <= x_max + EPSILON)
                return on_left_right or on_top_bottom

            # 両端点が枠上にあるかチェック
            if not (is_on_frame_boundary(x1, y1) and is_on_frame_boundary(x2, y2)):
                errors.append({
                    "type": "BoundaryVertexNotOnFrame",
                    "message": f"Boundary edge {i} has vertices not on frame boundary.",
                    "context": {
                        "edge_index": i,
                        "vertex_indices": [v1_idx, v2_idx],
                        "coords": [[x1, y1], [x2, y2]],
                        "frame_bounds": {"x": [x_min, x_max], "y": [y_min, y_max]}
                    }
                })
                continue

            # 条件1: 垂直な枠線か？（左右の境界線上）
            is_vertical_frame_edge = (
                math.isclose(x1, x2, rel_tol=EPSILON, abs_tol=EPSILON) and
                (math.isclose(x1, x_min, rel_tol=EPSILON, abs_tol=EPSILON) or 
                 math.isclose(x1, x_max, rel_tol=EPSILON, abs_tol=EPSILON))
            )

            # 条件2: 水平な枠線か？（上下の境界線上）
            is_horizontal_frame_edge = (
                math.isclose(y1, y2, rel_tol=EPSILON, abs_tol=EPSILON) and
                (math.isclose(y1, y_min, rel_tol=EPSILON, abs_tol=EPSILON) or 
                 math.isclose(y1, y_max, rel_tol=EPSILON, abs_tol=EPSILON))
            )

            # どちらの条件も満たさない場合はエラー
            if not (is_vertical_frame_edge or is_horizontal_frame_edge):
                errors.append({
                    "type": "BoundaryEdgeNotOnFrame",
                    "message": f"Boundary edge {i} ({v1_idx}-{v2_idx}) is not a valid horizontal or vertical segment on the frame.",
                    "context": {
                        "edge_index": i,
                        "vertex_indices": [v1_idx, v2_idx],
                        "coords": [[x1, y1], [x2, y2]],
                        "expected_frame": f"A vertical line on x={x_min} or x={x_max}, or a horizontal line on y={y_min} or y={y_max}",
                        "frame_bounds": {"x": [x_min, x_max], "y": [y_min, y_max]}
                    }
                })
    
    return errors if errors else None

def check_vertices_within_boundary(vertices_coords, x_range=(-200, 200), y_range=(-200, 200)):
    """
    全ての頂点が指定された境界内に存在するかを検証する。
    境界のデフォルト値は (-200, 200) に設定されています。
    """
    errors = []
    # x_rangeとy_rangeの最小値・最大値が正しい順序であることを確認
    x_min, x_max = min(x_range), max(x_range)
    y_min, y_max = min(y_range), max(y_range)

    for i, coord in enumerate(vertices_coords):
        # 座標がNoneや不正な形式でないことを確認
        if coord is None or len(coord) != 2:
            errors.append({
                "type": "InvalidCoordinateFormat",
                "vertex": i,
                "message": f"Vertex {i} has invalid coordinate format: {coord}.",
                "context": {"coords": coord}
            })
            continue
            
        x, y = coord
        # 判定ロジック: 許容誤差(EPSILON)を含めて範囲内かをチェック
        if not (x_min - EPSILON <= x <= x_max + EPSILON and
                y_min - EPSILON <= y <= y_max + EPSILON):
            errors.append({
                "type": "VertexOutOfBounds",
                "vertex": i,
                "message": f"Vertex {i} with coordinates ({x:.4f}, {y:.4f}) is outside the defined boundary.",
                "context": {
                    "coords": [x, y],
                    "x_boundary": [x_min, x_max],
                    "y_boundary": [y_min, y_max]
                }
            })
            
    return errors if errors else None


def get_connected_edges_assignments(vertex_index, edges_vertices, edges_assignment):
    """指定された頂点に接続するエッジの種類（山折り、谷折りなど）を取得する"""
    assignments = []
    for i, edge in enumerate(edges_vertices):
        if vertex_index in edge:
            assignments.append(edges_assignment[i])
    return assignments


# 頂点周りのエッジ情報を格納するためのデータ構造
HalfEdge = namedtuple('HalfEdge', [
    'vector_angle',      # 頂点から接続先へのベクトルの角度(atan2)
    'sector_angle',      # このエッジと次のエッジが成す角度(α)
    'assignment',        # "M", "V", "B" などの種類
    'edge_indices_tuple' # (中心頂点, 接続先頂点)
])


def get_ordered_half_edges(vertex_index, vertices_coords, edges_vertices, edges_assignment):
    """
    頂点に接続するハーフエッジを物理的にソートし、各セクターの角度と共に返す。
    """
    center_point = vertices_coords[vertex_index]
    
    # 頂点に接続するエッジとその接続先の情報を収集
    connected_edges = []
    for i, edge in enumerate(edges_vertices):
        if edge[0] == vertex_index:
            other_v_idx = edge[1]
            connected_edges.append({'assignment': edges_assignment[i], 'other_v': other_v_idx})
        elif edge[1] == vertex_index:
            other_v_idx = edge[0]
            connected_edges.append({'assignment': edges_assignment[i], 'other_v': other_v_idx})

    if len(connected_edges) < 2:
        return []

    # 各接続先へのベクトル角度(atan2)を計算
    sorted_edges_info = []
    for edge_info in connected_edges:
        p_coords = vertices_coords[edge_info['other_v']]
        angle = math.atan2(p_coords[1] - center_point[1], p_coords[0] - center_point[0])
        sorted_edges_info.append({
            'vector_angle': angle,
            'assignment': edge_info['assignment'],
            'other_v': edge_info['other_v']
        })
    
    # ベクトル角度で反時計回りにソート
    sorted_edges_info.sort(key=lambda x: x['vector_angle'])

    # HalfEdgeオブジェクトのリストを生成
    half_edges = []
    num_edges = len(sorted_edges_info)
    for i in range(num_edges):
        current_edge = sorted_edges_info[i]
        next_edge = sorted_edges_info[(i + 1) % num_edges]
        
        # 2つのベクトル間の角度（セクター角度）を計算
        sector_angle = next_edge['vector_angle'] - current_edge['vector_angle']
        if sector_angle < 0:
            sector_angle += 2 * math.pi
            
        half_edges.append(HalfEdge(
            vector_angle=current_edge['vector_angle'],
            sector_angle=sector_angle,
            assignment=current_edge['assignment'],
            edge_indices_tuple=(vertex_index, current_edge['other_v'])
        ))
        
    return half_edges


def check_boundary_count(assignments, vertex_index):
    """頂点に接続する境界線の数が0または2であるかを検証する"""
    b_count = assignments.count("B")
    if b_count not in [0, 2]:
        return {
            "type": "BoundaryCount",
            "vertex": vertex_index,
            "message": f"Vertex {vertex_index} has an invalid number of boundary edges. Expected 0 or 2, but found {b_count}.",
            "context": {
                "boundary_count": b_count,
                "expected_values": [0, 2]
            }
        }
    return None


def check_overlapping_creases(ordered_half_edges, vertex_index):
    """
    頂点から伸びる折り線が幾何学的に重複していないかを検証する。
    入力リストはベクトル角度でソート済みであることを前提とする。
    """
    num_edges = len(ordered_half_edges)
    if num_edges < 2:
        return None

    for i in range(num_edges):
        # 隣接するエッジを比較。リストは循環しているとみなす。
        current_he = ordered_half_edges[i]
        next_he = ordered_half_edges[(i + 1) % num_edges]

        if math.isclose(current_he.vector_angle, next_he.vector_angle, rel_tol=EPSILON, abs_tol=EPSILON):
            # 重複を検出
            return {
                "type": "OverlappingCreases",
                "vertex": vertex_index,
                "message": f"Vertex {vertex_index} has geometrically overlapping creases.",
                "context": {
                    "overlapping_angle_deg": math.degrees(current_he.vector_angle),
                    # 重複している線の先の頂点IDを取得
                    "involved_vertices": sorted([
                        current_he.edge_indices_tuple[1],
                        next_he.edge_indices_tuple[1]
                    ])
                }
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


def validate_fold_file(file_path):
    """
    .foldファイルを読み込み、トポロジー関連のルールを検証する。
    エッジ割り当て（M/V）に関するチェックは除外し、川崎の定理のみ実装。
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        return {"valid": False, "errors": [{"type": "File Error", "message": str(e)}]}

    required_keys = ["vertices_coords", "edges_vertices", "edges_assignment"]
    for key in required_keys:
        if key not in data:
            return {"valid": False, "errors": [{"type": "Format Error", "message": f"Missing required key: '{key}'"}]}

    vertices_coords = data["vertices_coords"]
    edges_vertices = data["edges_vertices"]
    edges_assignment = data["edges_assignment"]
    
    errors = []

    # --- 優先度1: 大域的・幾何学的健全性チェック ---
    global_errors = []
    
    # 頂点が境界内にあるかチェック
    boundary_errors = check_vertices_within_boundary(vertices_coords)
    if boundary_errors:
        global_errors.extend(boundary_errors)

    # 境界線が枠線上にあるかチェック
    if len(global_errors) < MAX_ERRORS_TO_REPORT:
        frame_errors = check_boundary_edges_on_frame(
            vertices_coords, edges_vertices, edges_assignment
        )
        if frame_errors:
            global_errors.extend(frame_errors)

    # エッジの不正な交差をチェック
    if len(global_errors) < MAX_ERRORS_TO_REPORT:
        intersection_errors = check_improper_edge_intersections(vertices_coords, edges_vertices)
        if intersection_errors:
            global_errors.extend(intersection_errors)
    
    if global_errors:
        # 大域的エラーが見つかった場合、それを上限まで追加し、即座に終了する
        errors.extend(global_errors)
        return {"valid": False, "errors": errors[:MAX_ERRORS_TO_REPORT]}

    # --- 優先度2: 頂点ごとのチェック ---
    for i in range(len(vertices_coords)):
        
        # 局所的・幾何学的健全性チェック
        local_geom_errors = []
        assignments = get_connected_edges_assignments(i, edges_vertices, edges_assignment)
        
        # 境界線の数をチェック
        err = check_boundary_count(assignments, i)
        if err: 
            local_geom_errors.append(err)

        # 重複する折り線をチェック
        ordered_half_edges = get_ordered_half_edges(i, vertices_coords, edges_vertices, edges_assignment)
        if ordered_half_edges:  # 接続辺がある場合のみチェック
            err = check_overlapping_creases(ordered_half_edges, i)
            if err: 
                local_geom_errors.append(err)

        if local_geom_errors:
            errors.extend(local_geom_errors)
            if len(errors) >= MAX_ERRORS_TO_REPORT:
                return {"valid": False, "errors": errors[:MAX_ERRORS_TO_REPORT]}
            continue

        # --- 川崎の定理チェック（境界頂点は除外） ---
        if ordered_half_edges and assignments.count("B") == 0:
            # 内部頂点のみ川崎の定理をチェック
            sector_angles = [he.sector_angle for he in ordered_half_edges]
            err = check_kawasaki_theorem(sector_angles, i)
            if err:
                errors.append(err)
                if len(errors) >= MAX_ERRORS_TO_REPORT:
                    return {"valid": False, "errors": errors[:MAX_ERRORS_TO_REPORT]}

    if errors:
        return {"valid": False, "errors": errors[:MAX_ERRORS_TO_REPORT]}
    
    return {"valid": True, "errors": []}


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python topology_generator_validator.py <path_to_fold_file>")
        sys.exit(1)
        
    file_path = sys.argv[1]
    result = validate_fold_file(file_path)
    print(json.dumps(result, indent=2))
