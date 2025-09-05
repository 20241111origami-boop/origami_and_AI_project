
import json
import math
import sys
from collections import defaultdict, namedtuple

# 浮動小数点数演算の許容誤差を定義
EPSILON = 1e-9

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
