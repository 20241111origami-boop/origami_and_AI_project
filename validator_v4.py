import json
import math
import sys
from collections import defaultdict, namedtuple


EPSILON = 1e-9
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

# --- ここから新しいコードを追加 ---

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


def check_big_little_big_theorem(ordered_half_edges, vertex_index):
    """
    Big-Little-Big定理から導かれる局所的な条件を検証する。
    ルール: 連続する3つのセクター角 a1, a2, a3 について、もし a1 > a2 かつ a3 > a2 ならば、
          a2 を形成する2本の折り線(e2, e3)の割り当ては異ならなければならない (M/V)。
    """
    num_edges = len(ordered_half_edges)
    if num_edges < 3:
        # 3辺未満の頂点では、この条件は適用できない
        return None

    for i in range(num_edges):
        # 連続する3つのセクター角を循環的に取得する
        # he_prev.sector_angle は a1 に相当
        # he_min.sector_angle  は a2 に相当
        # he_next.sector_angle は a3 に相当
        he_prev = ordered_half_edges[i]
        he_min  = ordered_half_edges[(i + 1) % num_edges]
        he_next = ordered_half_edges[(i + 2) % num_edges]

        angle1 = he_prev.sector_angle
        angle2 = he_min.sector_angle
        angle3 = he_next.sector_angle

        # 条件: angle1 > angle2 かつ angle3 > angle2 (angle2が局所最小角)
        # 浮動小数点数の比較のため、A > B を A - B > EPSILON でチェック
        if (angle1 - angle2 > EPSILON) and (angle3 - angle2 > EPSILON):
            # 条件を満たした場合、angle2を形成する2本の折り線の割り当てをチェックする。
            # angle2 (he_min.sector_angle) は、折り線 he_min と he_next によって形成される。
            assignment1 = he_min.assignment
            assignment2 = he_next.assignment

            # M(山折り)またはV(谷折り)でない場合はチェック対象外
            if assignment1 not in ["M", "V"] or assignment2 not in ["M", "V"]:
                continue

            # 割り当てが同じであればルール違反
            if assignment1 == assignment2:
                return {
                    "type": "BigLittleBigCondition",
                    "vertex": vertex_index,
                    "message": (
                        f"Vertex {vertex_index} fails the local minima condition (derived from Big-Little-Big). "
                        f"The two creases forming a local minimum angle must have different assignments (one mountain, one valley)."
                    ),
                    "context": {
                        "local_minimum_angle_deg": math.degrees(angle2),
                        "surrounding_angles_deg": [math.degrees(angle1), math.degrees(angle3)],
                        "conflicting_assignments": [assignment1, assignment2],
                        "involved_vertices": sorted([
                            he_min.edge_indices_tuple[1],
                            he_next.edge_indices_tuple[1]
                        ])
                    }
                }
    
    # 全ての局所最小角でルールが満たされた
    return None

# --- ここから新しいコードを追加 ---

def check_generalized_blb_lemma(ordered_half_edges, vertex_index):
    """
    一般化された大小大の補題（Generalized Big-Little-Big Lemma）を検証する。
    """
    # ステップ1: 適用除外条件の事前判定 (上位で実施済みと想定)
    # BORDERやUNASSIGNEDが含まれていないことを前提とする

    num_edges = len(ordered_half_edges)
    if num_edges < 3:
        return None # 3辺未満ではシーケンスが形成されない

    sector_angles = [he.sector_angle for he in ordered_half_edges]

    # ステップ3: 等角シーケンスの特定と適用可否の判定
    i = 0
    while i < num_edges:
        # 等角シーケンスを探す
        j = i
        while math.isclose(sector_angles[j], sector_angles[(j + 1) % num_edges], rel_tol=EPSILON, abs_tol=EPSILON):
            j = (j + 1) % num_edges
            if j == (i - 1 + num_edges) % num_edges: # 1周してしまった(全角が等しい)
                return None # この補題は適用されない

        if i == j: # シーケンス長が1ならスキップ
            i += 1
            continue

        # シーケンスSを特定
        # Pythonのスライスと違い、循環インデックスを扱う
        sequence_indices = []
        curr = i
        while True:
            sequence_indices.append(curr)
            if curr == j: break
            curr = (curr + 1) % num_edges
        
        # 適用前件(precondition)の確認
        alpha_s = sector_angles[i]
        alpha_prev_idx = (i - 1 + num_edges) % num_edges
        alpha_next_idx = (j + 1) % num_edges

        alpha_prev = sector_angles[alpha_prev_idx]
        alpha_next = sector_angles[alpha_next_idx]

        # 条件: α_prev > α_s AND α_next > α_s
        # 浮動小数点数なので (a > b) は (a - b >= EPSILON) で判定
        if (alpha_prev - alpha_s >= EPSILON) and (alpha_next - alpha_s >= EPSILON):
            # --- ステップ4: 補題の規則検証 ---
            
            # 1. 折り線本数の計数
            # k個の角度シーケンスは、k+1本の折り線で形成される
            edge_count = len(sequence_indices) + 1
            
            # 2. 山・谷の計数
            # 該当する k+1 本の折り線は、シーケンスの開始インデックス(i)から
            # 終了インデックス(j)の *次* まで
            mountain_count = 0
            valley_count = 0
            
            edge_sub_indices = []
            curr = i
            while True:
                edge_sub_indices.append(curr)
                if curr == alpha_next_idx: break # シーケンスを形成する最後のエッジ
                curr = (curr + 1) % num_edges

            # 実際のM/Vカウント
            for edge_idx in edge_sub_indices:
                assignment = ordered_half_edges[edge_idx].assignment
                if assignment == "M":
                    mountain_count += 1
                elif assignment == "V":
                    valley_count += 1

            # 3. 規則判定
            error = None
            if edge_count % 2 == 0: # 偶数本
                if mountain_count != valley_count:
                    error = "mountain_count == valley_count"
            else: # 奇数本
                if abs(mountain_count - valley_count) != 1:
                    error = "abs(mountain_count - valley_count) == 1"
            
            if error:
                # 違反が検出された
                return {
                    "type": "GeneralizedBigLittleBigLemma",
                    "vertex": vertex_index,
                    "message": f"Vertex {vertex_index}: " + (
                        f"An even-edge ({edge_count}) equal-angle sequence does not have the same number of mountains and valleys."
                        if edge_count % 2 == 0 else
                        f"An odd-edge ({edge_count}) equal-angle sequence does not have a mountain/valley count difference of 1."
                    ),
                    "context": {
                        "violating_sequence": [math.degrees(alpha_s)] * len(sequence_indices),
                        "edge_count": edge_count,
                        "mountain_count": mountain_count,
                        "valley_count": valley_count,
                        "expected_condition": error
                    }
                }
        
        # 次の探索開始位置へ
        if j < i: # 1周した場合
            break
        i = j + 1

    # ステップ5: 最終判定
    return None # 全てのシーケンスが規則を満たした

def validate_fold_file(file_path):
    """
    .foldファイルを読み込み、平坦折り畳み可能性のルールを検証する (GBLB対応版)
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
        # (変更なし) 頂点に接続するエッジの種類を取得
        assignments = get_connected_edges_assignments(i, edges_vertices, edges_assignment)
        
        # (変更なし) 境界("B")に接続する頂点は内点ではないため、定理のチェックをスキップ
        if "B" in assignments:
            continue
        
        # (変更なし) 前川の定理
        error = check_maekawa_theorem(assignments, i)
        if error:
            errors.append(error)

        # --- ▼ ここから変更箇所 ▼ ---

        # 新しいヘルパー関数で、ソート済みのエッジと角度の情報を取得
        ordered_half_edges = get_ordered_half_edges(i, vertices_coords, edges_vertices, edges_assignment)
        if not ordered_half_edges:
            continue
        
        # 各定理の検証には、この構造化されたデータから必要な情報を渡す
        sector_angles_rad = [he.sector_angle for he in ordered_half_edges]
            
        # 川崎の定理 (引数を変更)
        error = check_kawasaki_theorem(sector_angles_rad, i)
        if error:
            errors.append(error)
        
        # 大小大の定理 (引数を変更)
        error = check_big_little_big_theorem(sector_angles_rad, i)
        if error:
            errors.append(error)

        # 【新規】一般化された大小大の補題の検証
        error = check_generalized_blb_lemma(ordered_half_edges, i)
        if error:
            errors.append(error)
            
        # --- ▲ ここまで変更箇所 ▲ ---

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
