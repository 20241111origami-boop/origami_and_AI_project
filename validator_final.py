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


# 頂点周りのエッジ情報を格納するためのデータ構造
HalfEdge = namedtuple('HalfEdge', [
    'vector_angle',      # 頂点から接続先へのベクトルの角度(atan2)
    'sector_angle',      # このエッジと次のエッジが成す角度(α)
    'assignment',        # "M", "V", "B" などの種類
    'edge_indices_tuple' # (中心頂点, 接続先頂点)
])

# --- ここから新しいコードを追加 ---

class RingElement:
    """シミュレーションで扱う環状リストの要素"""
    def __init__(self, angle, line_type):
        self.angle = angle
        self.line_type = line_type

    def __repr__(self):
        return f"RingElement(angle={math.degrees(self.angle):.2f}, type='{self.line_type}')"

class FoldingRing:
    """折り畳みシミュレーションを管理する環状リスト"""
    def __init__(self, ordered_half_edges):
        # RingElementのリストを作成
        # 角度(sector_angle)と、その角度の *直後* の折り線の種類(assignment)をペアにする
        self.elements = []
        num_edges = len(ordered_half_edges)
        for i in range(num_edges):
            self.elements.append(RingElement(
                angle=ordered_half_edges[i].sector_angle,
                line_type=ordered_half_edges[(i + 1) % num_edges].assignment
            ))
        
        self.minimal_indices = self._find_all_minimal_indices()

    def size(self):
        return len(self.elements)

    def _find_all_minimal_indices(self):
        if not self.elements:
            return []
        min_angle = min(el.angle for el in self.elements)
        return [i for i, el in enumerate(self.elements) if math.isclose(el.angle, min_angle, rel_tol=EPSILON, abs_tol=EPSILON)]

    def pop_minimal_index(self):
        if not self.minimal_indices:
            return None
        return self.minimal_indices.pop(0)

    def fold_at(self, index):
        """
        指定されたインデックスで折り畳み操作を実行し、リストを更新する
        """
        n = self.size()
        if n <= 2: return

        # 最小角(m)とその両隣(p, n)を特定
        p_idx = (index - 1 + n) % n
        m_idx = index
        n_idx = (index + 1) % n
        
        p = self.elements[p_idx]
        m = self.elements[m_idx]
        n = self.elements[n_idx]

        # 新しい要素を計算 (Erik Demaineのアルゴリズムに基づく)
        # 角度は p + n - m、線の種類は p のものを継承
        new_angle = p.angle + n.angle - m.angle
        new_element = RingElement(new_angle, p.line_type)

        # 新しい環状リストを構築 (3要素を削除し、1要素を挿入)
        new_elements = []
        # p_idxがm_idxやn_idxより大きい場合(リストの先頭をまたぐ場合)のケア
        indices_to_remove = sorted([p_idx, m_idx, n_idx], reverse=True)
        
        temp_list = self.elements[:]
        # 後ろのインデックスから削除することで、前のインデックスがずれないようにする
        del temp_list[indices_to_remove[0]]
        del temp_list[indices_to_remove[1]]
        del temp_list[indices_to_remove[2]]
        
        # p_idx のあった場所に新しい要素を挿入
        # 削除によってp_idxの位置がずれる可能性を考慮
        insert_pos = p_idx - sum(1 for i in indices_to_remove if i < p_idx)
        temp_list.insert(insert_pos, new_element)
        
        self.elements = temp_list
        
        # 最小角リストを再計算
        self.minimal_indices = self._find_all_minimal_indices()
      
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


# --- ここから新しいコードを追加 ---
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

# --- ここから新しいコードを追加 ---
def check_folding_simulation(ordered_half_edges, vertex_index):
    """
    折り畳みシミュレーションを行い、頂点の平坦折り畳み可能性(十分条件)を検証する
    """
    ring = FoldingRing(ordered_half_edges)

    while ring.size() > 2:
        min_index = ring.pop_minimal_index()
        
        if min_index is None:
            return {
                "type": "FoldingSimulation",
                "vertex": vertex_index,
                "message": f"Vertex {vertex_index}: Folding simulation failed. No minimal angle found to continue.",
                "context": {
                    "failure_reason": "NO_MINIMAL_ANGLE_FOUND",
                    "final_state_size": ring.size(),
                    "final_state_details": [{"angle_deg": math.degrees(el.angle), "type": el.line_type} for el in ring.elements]
                }
            }
        
        ring.fold_at(min_index)

    # --- ステップ3: 最終判定 ---
    if ring.size() != 2:
        return {
            "type": "FoldingSimulation",
            "vertex": vertex_index,
            "message": f"Vertex {vertex_index}: Folding simulation failed. Final state does not have exactly 2 elements.",
            "context": {
                "failure_reason": "INVALID_FINAL_STATE",
                "final_state_size": ring.size(),
                "final_state_details": [{"angle_deg": math.degrees(el.angle), "type": el.line_type} for el in ring.elements]
            }
        }

    el1, el2 = ring.elements[0], ring.elements[1]

    if not math.isclose(el1.angle, el2.angle, rel_tol=EPSILON, abs_tol=EPSILON):
        return {
            "type": "FoldingSimulation",
            "vertex": vertex_index,
            "message": f"Vertex {vertex_index}: Folding simulation failed. Angles of the final two elements do not match.",
            "context": {
                "failure_reason": "FINAL_ANGLES_MISMATCH",
                "final_state_size": 2,
                "final_state_details": [{"angle_deg": math.degrees(el.angle), "type": el.line_type} for el in ring.elements]
            }
        }

    # --- ステップ4: 成功 ---
    # 幾何学的な検証が成功すればOK。line_typeの最終チェックは行わない。
    return None



def validate_fold_file(file_path):
    """
    .foldファイルを読み込み、平坦折り畳み可能性のルールを検証する (最終版)
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

    # 各頂点についてループ
    for i in range(len(vertices_coords)):
        # --- レベル0: 基本的な健全性チェック ---
        assignments = get_connected_edges_assignments(i, edges_vertices, edges_assignment)
        
        error = check_boundary_count(assignments, i)
        if error:
            errors.append(error)
            continue

        ordered_half_edges = get_ordered_half_edges(i, vertices_coords, edges_vertices, edges_assignment)
        if not ordered_half_edges:
            continue
        
        error = check_overlapping_creases(ordered_half_edges, i)
        if error:
            errors.append(error)
            continue # 健全性チェックに失敗した場合、以降の検証は無意味

        # 境界頂点(Bの数が2)の場合、内点向けの定理は適用しない
        if assignments.count("B") > 0:
            continue

        # --- レベル1: 内点向けの必要条件チェック ---
        level1_errors = []
        level1_errors.append(check_maekawa_theorem(assignments, i))
        
        sector_angles_rad = [he.sector_angle for he in ordered_half_edges]
        level1_errors.append(check_kawasaki_theorem(sector_angles_rad, i))
        level1_errors.append(check_big_little_big_theorem(sector_angles_rad, i))
        level1_errors.append(check_generalized_blb_lemma(ordered_half_edges, i))
        
        level1_errors = [e for e in level1_errors if e is not None]
        if level1_errors:
            errors.extend(level1_errors)
            continue # レベル1でエラーがあれば、レベル2は実行しない

        # --- レベル2: 内点向けの十分条件チェック ---
        error = check_folding_simulation(ordered_half_edges, i)
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
