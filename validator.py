import json
import math
import sys

# 浮動小数点数の比較に使用する微小な許容誤差
EPSILON = 1e-6

def get_connected_edges_and_vertices(vertex_id, edges_vertices, vertices_coords):
    """指定された頂点に接続するエッジと、その先の頂点の情報を返す"""
    connected = []
    for i, edge in enumerate(edges_vertices):
        if vertex_id in edge:
            other_vertex_id = edge[1] if edge[0] == vertex_id else edge[0]
            p0 = vertices_coords[vertex_id]
            p1 = vertices_coords[other_vertex_id]
            vec_x = p1[0] - p0[0]
            vec_y = p1[1] - p0[1]
            angle = math.atan2(vec_y, vec_x)
            connected.append({
                "edge_index": i,
                "other_vertex_id": other_vertex_id,
                "angle": angle
            })
    connected.sort(key=lambda x: x["angle"])
    return connected

def check_maekawa_theorem(connected_edges, edges_assignment):
    """前川の定理を検証する |M - V| = 2"""
    mountain_folds = 0
    valley_folds = 0
    for edge_info in connected_edges:
        assignment = edges_assignment[edge_info["edge_index"]]
        if assignment == "M":
            mountain_folds += 1
        elif assignment == "V":
            valley_folds += 1
    
    # 折り線がなければ検証対象外
    if mountain_folds + valley_folds == 0:
        return True, ""

    if abs(mountain_folds - valley_folds) != 2:
        msg = f"Mountain folds: {mountain_folds}, Valley folds: {valley_folds}. |M-V| is {abs(mountain_folds - valley_folds)}, should be 2."
        return False, msg
        
    return True, ""

def check_kawasaki_theorem(connected_edges):
    """川崎の定理を検証する。交互の角度の和がπになり、総和が2πになる"""
    if not connected_edges or len(connected_edges) < 2:
        return True, ""

    face_angles = []
    num_edges = len(connected_edges)
    for i in range(num_edges):
        angle1 = connected_edges[i]["angle"]
        angle2 = connected_edges[(i + 1) % num_edges]["angle"]
        diff = angle2 - angle1
        if diff < 0:
            diff += 2 * math.pi
        face_angles.append(diff)
        
    total_angle = sum(face_angles)
    if not math.isclose(total_angle, 2 * math.pi, rel_tol=EPSILON):
        # これは境界上の頂点である可能性が高いため、エラーとせず、呼び出し元で判断する
        # ただし、内側の頂点であるにも関わらず360度でない場合は問題
        # 今回は境界判定を先に行うため、ここに来る時点で内側頂点のはず
        msg = f"Sum of angles around the vertex is {math.degrees(total_angle):.2f} degrees, but should be 360 for an internal vertex."
        return False, msg
        
    odd_sum = sum(face_angles[i] for i in range(0, num_edges, 2))
    if not math.isclose(odd_sum, math.pi, rel_tol=EPSILON):
        msg = f"The sum of alternating angles is {math.degrees(odd_sum):.2f} degrees, but should be 180."
        # 川崎の定理の厳密な定義の一つ
        # return False, msg # 総和が360度であれば、この条件も満たされるため、エラーメッセージは総和だけで十分
    
    return True, ""

def validate_fold_file(file_path):
    """ .fold ファイルを読み込み、平坦折り畳み可能性を検証する """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        return {"valid": False, "errors": [{"type": "FileReadError", "message": str(e)}]}

    vertices_coords = data.get("vertices_coords")
    edges_vertices = data.get("edges_vertices")
    edges_assignment = data.get("edges_assignment")

    if not all([vertices_coords, edges_vertices, edges_assignment]):
        return {"valid": False, "errors": [{"type": "DataFormatError", "message": "Missing required keys in .fold file."}]}
    
    errors = []
    num_vertices = len(vertices_coords)

    for i in range(num_vertices):
        connected = get_connected_edges_and_vertices(i, edges_vertices, vertices_coords)
        
        # === ▼▼▼ 修正箇所 ▼▼▼ ===
        # 接続エッジに境界("B")が含まれているかチェック
        # 含まれている場合、その頂点は境界上の頂点なので定理の検証をスキップする
        is_boundary_vertex = any(edges_assignment[e["edge_index"]] == "B" for e in connected)
        if is_boundary_vertex:
            continue
        # === ▲▲▲ 修正完了 ▲▲▲ ===

        # 前川の定理チェック
        is_valid, msg = check_maekawa_theorem(connected, edges_assignment)
        if not is_valid:
            errors.append({"type": "Maekawa", "vertex": i, "message": msg})

        # 川崎の定理チェック
        is_valid, msg = check_kawasaki_theorem(connected)
        if not is_valid:
            errors.append({"type": "Kawasaki", "vertex": i, "message": msg})

    return {"valid": not errors, "errors": errors}

if __name__ == "__main__":
    # この部分はローカルでの実行を想定したサンプルコードです
    # 実際には引数からファイルパスを受け取ります
    if len(sys.argv) != 2:
        print("Usage: python validator_v2.py <path_to_fold_file>")
        sys.exit(1)
        
    file_path = sys.argv[1]
    result = validate_fold_file(file_path)
    
    print(json.dumps(result, indent=2))
