
from pathlib import Path

import csv, math, time, random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

reward_tolerance = 1e-4
beta_a_default = 0.3

# Параметры по умолчанию (используются при AUTO_TUNE = False)
DEFAULT_C = 1.414
DEFAULT_MAX_DEPTH = None
DEFAULT_COV_THRESH = 0.95
DEFAULT_GAMMA_COV = 1.0
DEFAULT_ITERATIONS = 500
DEFAULT_ALPHA = 0.5

# 1. Получаем абсолютный путь к папке, где лежит этот скрипт (src/)
CURRENT_DIR = Path(__file__).parent.resolve()
# 2. Поднимаемся на уровень выше (в корень проекта) и заходим в папку data/
DATA_DIR = CURRENT_DIR.parent / "data"
# 3. Формируем полный путь к файлу
CSV_FILE_PATH = DATA_DIR / "data.csv"

SRC_DIR = Path(__file__).resolve().parent        # /src
RESULTS_DIR = SRC_DIR.parent / 'results'          # /results (рядом с /src)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)     # создаёт /results если нет


# ====== ВКЛ/ВЫКЛ АВТОАНАЛИЗ И ПОДБОР ПАРАМЕТРОВ ======
AUTO_TUNE = True #False   # True = вкл автоанализ и подбор, False = параметры по умолчанию


# =====================================================================
# 1. ОСНОВНЫЕ ФУНКЦИИ
# =====================================================================

def _results_path(filename):
    p = Path(filename)
    if p.is_absolute() or len(p.parts) > 1:
        return str(p)
    return str(RESULTS_DIR / filename)

def custom_sort_key(x):
    coverage = x.get('coverage', 0.0)
    return (-x['ku'][0], x['ku'][1], -x['ku'][2], x['ku'][3], coverage)

def build_obj(triplets):
    graph = {}
    for state, action, next_state in triplets:
        if state not in graph:
            graph[state] = [[[], action]]
        actions_list = [item[1] for item in graph[state]]
        if action not in actions_list:
            graph[state].append([[next_state], action])
        else:
            idx = actions_list.index(action)
            if next_state not in graph[state][idx][0]:
                graph[state][idx][0].append(next_state)
    return graph

def ku_get(graph):
    n = 0; u = 0; actions = []
    for s in graph:
        for t in graph[s]:
            if t[0] != []:
                u += len(t[0]) - 1
                n += 1
                actions.append(t[1])
    if n != 0:
        return (round(1 - u / (n + u), 3),
                round(len(set(actions)) / (n + u), 3),
                n + u,
                len(set(actions)))
    return None

def compute_coverage(n_triplets, n_rows):
    """Coverage в [0, 1]: |sas'| / N"""
    if n_rows <= 0:
        return 0.0
    return round(n_triplets / n_rows, 4)

def build_graph_from_hypothesis(data, state_idx, action_idx):
    triplets = []
    for i in range(len(data) - 1):
        try:
            state = tuple(int(data[i][j]) for j in state_idx)
            action = tuple(int(data[i][k]) for k in action_idx)
            next_state = tuple(int(data[i+1][l]) for l in state_idx)
            if len(state_idx) > 0 and len(action_idx) > 0:
                triplets.append((state, action, next_state))
        except (ValueError, IndexError):
            continue
    if not triplets:
        return {}
    return build_obj(triplets)

def evaluate_hypothesis(data, state_idx, action_idx, n_rows,
                        alpha, beta_a, max_idx,
                        coverage_threshold, gamma_cov):
    """Возвращает (ku, norm_a, n_tr, n_a, coverage, reward)."""
    # Пустой state или action — гипотеза невалидна
    if not state_idx or not action_idx:
        return (0.0, 0.0, 0, 0, 0.0, 0.0)
    graph = build_graph_from_hypothesis(data, state_idx, action_idx)
    result = ku_get(graph)
    if result is None:
        return (0.0, 0.0, 0, 0, 0.0, 0.0)
    ku, norm_a, n_tr, n_a = result
    coverage = compute_coverage(n_tr, n_rows)
    if coverage >= coverage_threshold:
        return (ku, norm_a, n_tr, n_a, coverage, 0.0)
    na_cols = len(action_idx) / max_idx if max_idx > 0 else 0
    reward = ku - gamma_cov * coverage * coverage - alpha * norm_a - beta_a * na_cols
    return (ku, norm_a, n_tr, n_a, coverage, reward)

# =====================================================================
# 2. АВТОАНАЛИЗ (только autocorr + consistency, без кардинальности)
# =====================================================================

def analyze_data_structure(data, max_idx):
    n_rows = len(data)
    sample_size = min(n_rows - 1, 2000)
    step = max(1, (n_rows - 1) // sample_size)
    col_info = []
    for col in range(max_idx):
        values, next_values = [], []
        for i in range(0, n_rows - 1, step):
            try:
                v = int(data[i][col]); nv = int(data[i+1][col])
                values.append(v); next_values.append(nv)
            except (ValueError, IndexError):
                continue
        transitions = {}
        for v, nv in zip(values, next_values):
            if v not in transitions: transitions[v] = {}
            transitions[v][nv] = transitions[v].get(nv, 0) + 1
        consistency = 0.0
        if transitions:
            total = sum(sum(n.values()) for n in transitions.values())
            consistent = sum(max(n.values()) for n in transitions.values())
            consistency = consistent / total if total > 0 else 0.0
        n = len(values)
        if n > 1:
            mean_v = sum(values) / n; mean_nv = sum(next_values) / n
            cov = sum((v - mean_v) * (nv - mean_nv) for v, nv in zip(values, next_values)) / n
            std_v = (sum((v - mean_v) ** 2 for v in values) / n) ** 0.5
            std_nv = (sum((nv - mean_nv) ** 2 for nv in next_values) / n) ** 0.5
            autocorr = cov / (std_v * std_nv) if std_v > 0 and std_nv > 0 else 0.0
        else:
            autocorr = 0.0
        col_info.append({'col': col, 'consistency': round(consistency, 3), 'autocorr': round(autocorr, 3)})

    state_cols, action_cols, noise_cols = [], [], []
    for info in col_info:
        if info['autocorr'] > 0.3:
            state_cols.append(info['col']); info['class'] = 'STATE'
        elif info['autocorr'] < 0.15 and info['consistency'] > 0.15:
            action_cols.append(info['col']); info['class'] = 'ACTION'
        else:
            noise_cols.append(info['col']); info['class'] = 'NOISE'
    return {'state_cols': state_cols, 'action_cols': action_cols,
            'noise_cols': noise_cols, 'expected_depth': len(state_cols) + len(action_cols),
            'col_info': col_info}

def recommend_c_formula(max_idx, iterations, expected_depth):
    B = 2 * max_idx
    if B <= 1: return 1.414
    D_reach = math.log(max(iterations, 2)) / math.log(B)
    if expected_depth <= D_reach: return 1.414
    ratio = D_reach / expected_depth
    return round(max(1.414 * (ratio ** 3), 0.01), 4)

def auto_coverage_threshold(data, analysis, n_rows):
    state_cols = analysis['state_cols']
    action_cols = analysis['action_cols']
    if not state_cols or not action_cols:
        return 0.5
    state_space = 1
    for s in state_cols:
        vals = set()
        for row in range(min(n_rows, 500)):
            try: vals.add(int(data[row][s]))
            except: pass
        state_space *= max(len(vals), 2)
    action_space = 1
    for a in action_cols:
        vals = set()
        for row in range(min(n_rows, 500)):
            try: vals.add(int(data[row][a]))
            except: pass
        action_space *= max(len(vals), 2)
    expected_sas = min(n_rows, state_space * action_space)
    expected_cov = expected_sas / n_rows
    if expected_cov < 0.2: threshold = 0.3
    elif expected_cov < 0.5: threshold = expected_cov * 1.5
    else: threshold = min(0.95, expected_cov * 1.1)
    return round(threshold, 3)

# =====================================================================
# 3. ROLLOUT (оптимизированный — на подвыборке)
# =====================================================================

def greedy_rollout(data, state_idx, action_idx, max_idx, n_rows,
                   alpha, beta_a, coverage_threshold, gamma_cov,
                   max_rollout_steps, analysis_state, analysis_action):
    sub_n = min(n_rows, 500)
    sub_data = data[:sub_n]

    def eval_sub(s_idx, a_idx):
        '''оценивает гипотезу на подвыборке'''
        if not s_idx or not a_idx:
            return (0.0, 0.0, 0, 0, 0.0, 0.0)
        g = build_graph_from_hypothesis(sub_data, s_idx, a_idx)
        r = ku_get(g)
        if r is None:
            return (0.0, 0.0, 0, 0, 0.0, 0.0)
        ku, norm_a, n_tr, n_a = r
        cov = n_tr / sub_n
        if cov >= coverage_threshold:
            return (ku, norm_a, n_tr, n_a, cov, 0.0)
        na_cols = len(a_idx) / max_idx if max_idx > 0 else 0
        rew = ku - gamma_cov * cov * cov - alpha * norm_a - beta_a * na_cols
        return (ku, norm_a, n_tr, n_a, cov, rew)

    def eval_full(s_idx, a_idx):
        '''финальная валидация на полных данных'''
        return evaluate_hypothesis(data, s_idx, a_idx, n_rows,
                                    alpha, beta_a, max_idx,
                                    coverage_threshold, gamma_cov)

    best_state = list(state_idx)
    best_action = list(action_idx)
    ku, norm_a, n_tr, n_a, cov, rew = eval_sub(best_state, best_action)
    best_reward = rew

    # Кандидат 1: гипотеза из анализа
    if analysis_state and analysis_action:
        a_s = sorted(set(analysis_state + state_idx))
        a_a = sorted(set(analysis_action + action_idx))
        if len(set(a_s) | set(a_a)) == len(a_s) + len(a_a):
            r2 = eval_sub(a_s, a_a)
            if r2[5] > best_reward:
                best_reward = r2[5]
                best_state, best_action = a_s, a_a
                ku, norm_a, n_tr, n_a, cov = r2[0], r2[1], r2[2], r2[3], r2[4]

    # Кандидат 2: batch всех action
    if analysis_action:
        b_s = sorted(set(state_idx + analysis_state[:1]))
        b_a = sorted(set(analysis_action + action_idx))
        if len(set(b_s) | set(b_a)) == len(b_s) + len(b_a):
            r3 = eval_sub(b_s, b_a)
            if r3[5] > best_reward:
                best_reward = r3[5]
                best_state, best_action = b_s, b_a
                ku, norm_a, n_tr, n_a, cov = r3[0], r3[1], r3[2], r3[3], r3[4]

    # Кандидат 3: жадное добавление по одной фиче
    cur_s = list(best_state)
    cur_a = list(best_action)
    cur_rew = best_reward
    for _ in range(max_rollout_steps):
        candidates = []
        used = set(cur_s) | set(cur_a)
        for i in range(max_idx):
            if i not in used:
                ns = sorted(cur_s + [i]); na = sorted(cur_a)
                r1 = eval_sub(ns, na)
                candidates.append((r1[5], r1, ns, na))
                ns2 = sorted(cur_s); na2_ = sorted(cur_a + [i])
                r2 = eval_sub(ns2, na2_)
                candidates.append((r2[5], r2, ns2, na2_))
        if not candidates:
            break
        best_c = max(candidates, key=lambda c: c[0])
        if best_c[0] <= cur_rew + 1e-6:
            break
        cur_rew = best_c[0]
        cur_s, cur_a = best_c[2], best_c[3]
        r = best_c[1]
        ku, norm_a, n_tr, n_a, cov = r[0], r[1], r[2], r[3], r[4]

    if cur_rew > best_reward:
        best_reward = cur_rew
        best_state, best_action = cur_s, cur_a

    # Финальная валидация на полных данных
    ku_f, na_f, n_f, na_f_, cov_f, rew_f = eval_full(best_state, best_action)
    if rew_f > 0:
        best_reward = rew_f
        ku, norm_a, n_tr, n_a, cov = ku_f, na_f, n_f, na_f_, cov_f

    return best_state, best_action, best_reward, ku, norm_a, n_tr, n_a, cov

# =====================================================================
# 4. MCTS
# =====================================================================

class HypothesisNode:
    def __init__(self, state_idx, action_idx, parent=None, action_taken=None):
        self.state_idx = list(state_idx)
        self.action_idx = list(action_idx)
        self.parent = parent
        self.action_taken = action_taken
        self.children = []
        self.visits = 0
        self.wins = 0.0
        self.untried_actions = None
        self.depth = 0

    def get_untried_actions(self, max_idx, allowed_cols=None, max_total_cols=None):
        if self.untried_actions is not None:
            return self.untried_actions
        used = set(self.state_idx) | set(self.action_idx)
        if max_total_cols is not None and len(used) >= max_total_cols:
            self.untried_actions = []
            return self.untried_actions
        actions = []
        cols = allowed_cols if allowed_cols is not None else range(max_idx)
        for i in cols:
            if i not in used:
                actions.append(('add_state', i))
                actions.append(('add_action', i))
        self.untried_actions = actions
        return actions

    def is_fully_expanded(self):
        return len(self.untried_actions or []) == 0

    def expand(self, action, max_idx, allowed_cols=None, max_total_cols=None):
        new_state_idx = list(self.state_idx)
        new_action_idx = list(self.action_idx)
        if action[0] == 'add_state':
            new_state_idx.append(action[1])
        elif action[0] == 'add_action':
            new_action_idx.append(action[1])
        if self.untried_actions is not None and action in self.untried_actions:
            self.untried_actions.remove(action)
        child = HypothesisNode(new_state_idx, new_action_idx, parent=self, action_taken=action)
        child.depth = self.depth + 1
        child.get_untried_actions(max_idx, allowed_cols, max_total_cols)
        self.children.append(child)
        return child

    def ucb1(self, exploration_param=1.414):
        if self.visits == 0:
            return float('inf')
        exploitation = self.wins / self.visits
        exploration = exploration_param * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

    def best_child(self, exploration_param=1.414):
        return max(self.children, key=lambda c: c.ucb1(exploration_param))

class HypothesisMCTS:
    def __init__(self, max_idx=5, exploration_param=1.414,
                 beta_a=beta_a_default, max_depth=None, gamma_cov=3.0):
        self.max_idx = max_idx
        self.exploration_param = exploration_param
        self.beta_a = beta_a
        self.max_depth = max_depth
        self.gamma_cov = gamma_cov

    def search(self, data, iterations=500, min_triplets=5, alpha=0.5,
               coverage_threshold=0.5,
               init_state_idx=None, init_action_idx=None,
               analysis_state=None, analysis_action=None,
               max_total_cols=None):
        n_rows = len(data)
        init_state_idx = list(init_state_idx) if init_state_idx else []
        init_action_idx = list(init_action_idx) if init_action_idx else []

        # Подвыборка для быстрого rollout
        sub_n = min(n_rows, 2000)
        sub_data = data[:sub_n]

        allowed_cols = sorted(set((analysis_state or []) + (analysis_action or [])))
        if not allowed_cols:
            allowed_cols = list(range(self.max_idx))
        root = HypothesisNode(state_idx=init_state_idx, action_idx=init_action_idx)
        root.get_untried_actions(self.max_idx, allowed_cols, max_total_cols)
        best_hypothesis = None
        best_reward = None
        best_sort_key = None

        top_hypotheses = []
        TOP_N = 5

        def add_to_top(hyp_dict, rew):
            key = (tuple(sorted(hyp_dict['state_idx'])), tuple(sorted(hyp_dict['action_idx'])))
            existing = {(tuple(sorted(h['state_idx'])), tuple(sorted(h['action_idx'])))
                        for h in top_hypotheses}
            if key not in existing:
                h = dict(hyp_dict)
                h['reward'] = rew
                top_hypotheses.append(h)
                top_hypotheses.sort(key=lambda h: -h['reward'])
                del top_hypotheses[TOP_N:]

        def eval_sub(s_idx, a_idx):
            if not s_idx or not a_idx:
                return (0.0, 0.0, 0, 0, 0.0, 0.0)
            g = build_graph_from_hypothesis(sub_data, s_idx, a_idx)
            r = ku_get(g)
            if r is None:
                return (0.0, 0.0, 0, 0, 0.0, 0.0)
            ku, norm_a, n_tr, n_a = r
            cov = n_tr / sub_n
            if cov >= coverage_threshold:
                return (ku, norm_a, n_tr, n_a, cov, 0.0)
            na_cols = len(a_idx) / self.max_idx if self.max_idx > 0 else 0
            rew = ku - self.gamma_cov * cov * cov - alpha * norm_a - self.beta_a * na_cols
            return (ku, norm_a, n_tr, n_a, cov, rew)

        # Предвычисляем гипотезу из анализа
        analysis_hyp_reward = -1
        analysis_hyp = None
        if analysis_state and analysis_action:
            r = eval_sub(sorted(analysis_state), sorted(analysis_action))
            if r[5] > 0:
                analysis_hyp_reward = r[5]
                analysis_hyp = (sorted(analysis_state), sorted(analysis_action), r)

        for _ in range(iterations):
            node = root
            # 1. SELECTION
            while node.is_fully_expanded() and node.children:
                node = node.best_child(self.exploration_param)
            # 2. EXPANSION
            if not node.is_fully_expanded():
                if self.max_depth is not None and node.depth >= self.max_depth:
                    pass
                else:
                    action = random.choice(node.untried_actions)
                    node = node.expand(action, self.max_idx, allowed_cols, max_total_cols)

            # 3. SIMULATION — rollout на подвыборке
            # Текущий узел
            r_cur = eval_sub(node.state_idx, node.action_idx)
            reward = r_cur[5]

            # Rollout: добавляем фичи жадно (на подвыборке)
            roll_s = list(node.state_idx)
            roll_a = list(node.action_idx)
            roll_rew = reward
            roll_r = r_cur

            for _ in range(3):
                if max_total_cols is not None and len(roll_s) + len(roll_a) >= max_total_cols:
                    break
                candidates = []
                used = set(roll_s) | set(roll_a)
                for i in allowed_cols:
                    if i not in used:
                        ns = sorted(roll_s + [i]); na = sorted(roll_a)
                        r1 = eval_sub(ns, na)
                        candidates.append((r1[5], r1, ns, na))
                        ns2 = sorted(roll_s); na2 = sorted(roll_a + [i])
                        r2 = eval_sub(ns2, na2)
                        candidates.append((r2[5], r2, ns2, na2))
                if not candidates:
                    break
                best_c = max(candidates, key=lambda c: c[0])
                if best_c[0] <= roll_rew + 1e-6:
                    break
                roll_rew = best_c[0]
                roll_s, roll_a = best_c[2], best_c[3]
                roll_r = best_c[1]

            if roll_rew > reward:
                reward = roll_rew

            # Обновляем лучшую гипотезу (пустые state/action — не учитываем)
            if roll_s and roll_a and roll_r[2] >= min_triplets and roll_rew > 0:
                current_h = {
                    'state_idx': list(roll_s),
                    'action_idx': list(roll_a),
                    'ku': (roll_r[0], roll_r[1], roll_r[2], roll_r[3]),
                    'coverage': roll_r[4]
                }
                current_key = custom_sort_key(current_h)
                add_to_top(current_h, roll_rew)
                if (best_reward is None or roll_rew > best_reward or
                    (abs(roll_rew - best_reward) < reward_tolerance and current_key < best_sort_key)):
                    best_reward = roll_rew
                    best_sort_key = current_key
                    best_hypothesis = current_h

            # 4. BACKPROPAGATION
            bp = node
            while bp is not None:
                bp.visits += 1
                bp.wins += reward
                bp = bp.parent

        # Автоанализ — только как запасной вариант (MCTS не нашёл ничего)
        if analysis_hyp and analysis_hyp_reward > 0 and best_reward is None:
            a_s, a_a, a_r = analysis_hyp
            # Валидируем на полных данных
            full_r = evaluate_hypothesis(data, a_s, a_a, n_rows,
                alpha, self.beta_a, self.max_idx, coverage_threshold, self.gamma_cov)
            if full_r[5] > 0:
                best_hypothesis = {
                    'state_idx': list(a_s),
                    'action_idx': list(a_a),
                    'ku': (full_r[0], full_r[1], full_r[2], full_r[3]),
                    'coverage': full_r[4]
                }
                best_reward = full_r[5]

        # Валидируем лучшую на полных данных
        if best_hypothesis:
            S = best_hypothesis['state_idx']
            A = best_hypothesis['action_idx']
            full_r = evaluate_hypothesis(data, S, A, n_rows,
                alpha, self.beta_a, self.max_idx, coverage_threshold, self.gamma_cov)
            if full_r[5] > 0:
                best_hypothesis['ku'] = (full_r[0], full_r[1], full_r[2], full_r[3])
                best_hypothesis['coverage'] = full_r[4]
                best_hypothesis['reward'] = full_r[5]

        # Валидируем топ-N на полных данных
        validated_top = []
        for h in top_hypotheses:
            S = h['state_idx']
            A = h['action_idx']
            full_r = evaluate_hypothesis(data, S, A, n_rows,
                alpha, self.beta_a, self.max_idx, coverage_threshold, self.gamma_cov)
            if full_r[5] > 0:
                h['ku'] = (full_r[0], full_r[1], full_r[2], full_r[3])
                h['coverage'] = full_r[4]
                h['reward'] = full_r[5]
                validated_top.append(h)
        validated_top.sort(key=lambda h: -h['reward'])
        validated_top = validated_top[:TOP_N]

        # Сравниваем лучшую гипотезу с топом после валидации на полных данных
        if validated_top:
            top_best = validated_top[0]
            if (not best_hypothesis or
                top_best.get('reward', 0) > best_hypothesis.get('reward', 0)):
                best_hypothesis = top_best

        return best_hypothesis, validated_top

# =====================================================================
# 5. АВТОПОДБОР
# =====================================================================

def auto_tune(data, max_idx, iterations, alpha=0.5, beta_a=beta_a_default):
    print("\n" + "=" * 70)
    print("АВТОАНАЛИЗ И ПОДБОР ПАРАМЕТРОВ")
    print("=" * 70)
    analysis = analyze_data_structure(data, max_idx)
    expected_depth = analysis['expected_depth']
    c_formula = recommend_c_formula(max_idx, iterations, expected_depth)
    max_depth = expected_depth + 1 if expected_depth > 0 else None
    cov_threshold = auto_coverage_threshold(data, analysis, len(data))
    gamma_cov = round(0.3 / (cov_threshold * cov_threshold), 2) if cov_threshold > 0 else 3.0

    print(f"\nАнализ данных:")
    print(f"  Всего колонок      : {max_idx}")
    print(f"  Фактор ветвления B : {2 * max_idx}")
    print(f"  State (оценка)     : {len(analysis['state_cols'])} {analysis['state_cols']}")
    print(f"  Action (оценка)    : {len(analysis['action_cols'])} {analysis['action_cols']}")
    print(f"  Noise (оценка)     : {len(analysis['noise_cols'])} {analysis['noise_cols']}")
    print(f"  Ожидаемая глубина  : {expected_depth}")
    print(f"  max_depth          : {max_depth}")
    B = 2 * max_idx
    D_reach = math.log(max(iterations, 2)) / math.log(max(B, 2))
    print(f"  D_reachable (T={iterations}) : {D_reach:.2f}")
    print(f"  c_formula          : {c_formula}")
    print(f"  cov_threshold      : {cov_threshold}")
    print(f"  gamma_cov          : {gamma_cov}")
    print(f"\n  Колонки:")
    for info in analysis['col_info']:
        print(f"    col {info['col']}: consist={info['consistency']}, autocorr={info['autocorr']} -> {info['class']}")
    print("=" * 70)
    return c_formula, max_depth, analysis, cov_threshold, gamma_cov

# =====================================================================
# 6. ЗАГРУЗКА И ГРАФИКИ
# =====================================================================

def load_real_data(filename='data.csv'):
    data = []
    try:
        with open(filename, 'r', newline='', encoding='utf-8') as file:
            for row in csv.reader(file):
                if row: data.append(row)
        print(f"Загружено {len(data)} строк из {filename}")
        return data
    except FileNotFoundError:
        print(f"Файл {filename} не найден")
        return []

def evaluate_parameter(data, param_name, param_values, fixed_params,
                       max_idx, beta_a=beta_a_default, max_depth=None,
                       gamma_cov=3.0, cov_threshold=0.5,
                       analysis_state=None, analysis_action=None):
    results = []
    for val in param_values:
        kwargs = {**fixed_params, param_name: val}
        mcts = HypothesisMCTS(max_idx=max_idx,
                             exploration_param=kwargs.get('exploration_param', 1.414),
                             beta_a=beta_a, max_depth=max_depth, gamma_cov=gamma_cov)
        start = time.time()
        hyp, _top = mcts.search(data, iterations=kwargs.get('iterations', 500),
                          min_triplets=kwargs.get('min_triplets', 5),
                          alpha=kwargs.get('alpha', 0.5),
                          coverage_threshold=kwargs.get('coverage_threshold', cov_threshold),
                          analysis_state=analysis_state, analysis_action=analysis_action)
        elapsed = time.time() - start
        if hyp and hyp['state_idx'] and hyp['action_idx']:
            ku = hyp['ku'][0]; norm_a = hyp['ku'][1]
            n_sas = hyp['ku'][2]; n_a = hyp['ku'][3]
            cov = hyp.get('coverage', 0.0)
            S = hyp['state_idx']; A = hyp['action_idx']
            na_cols = len(A) / max_idx if max_idx > 0 else 0
            reward = ku - gamma_cov * cov * cov - kwargs.get('alpha', 0.5) * norm_a - beta_a * na_cols
        else:
            ku, norm_a, n_sas, n_a, cov, reward = 0, 0, 0, 0, 0, 0
            S, A = [], []
        results.append({'val': val, 'ku': ku, 'norm_a': norm_a, 'n_sas': n_sas,
                        'n_a': n_a, 'coverage': cov, 'reward': reward,
                        'time': elapsed, 'state_idx': S, 'action_idx': A})
        print(f"  {param_name}={val}: (ku={ku}, norm_a={norm_a}, |sas'|={n_sas}, "
              f"|a|={n_a}, cov={cov}, reward={reward:.4f}) time={elapsed:.2f}s | S={S}, A={A}")
    return results

def plot_results(data, max_idx, c_rec=None, max_depth=None,
                 beta_a=beta_a_default, gamma_cov=3.0, cov_threshold=0.5,
                 analysis_state=None, analysis_action=None):
    plt.figure(figsize=(18, 10))
    plt.style.use('seaborn-v0_8-darkgrid')

    print(f"\n1. Оценка итераций (c={c_rec}, alpha=0.5)...")
    it_vals = [10, 50, 100, 200, 500, 700]
    it_res = evaluate_parameter(data, 'iterations', it_vals,
        {'exploration_param': c_rec, 'min_triplets': 5, 'coverage_threshold': cov_threshold},
        max_idx, beta_a=beta_a, max_depth=max_depth, gamma_cov=gamma_cov,
        cov_threshold=cov_threshold, analysis_state=analysis_state, analysis_action=analysis_action)

    ax1 = plt.subplot(2, 3, 1)
    l1, = ax1.plot(it_vals, [r['ku'] for r in it_res], marker='o', color='b', label='ku', linewidth=2)
    ax1.set_xlabel('Итерации'); ax1.set_ylabel('ku', color='b'); ax1.tick_params(axis='y', labelcolor='b')
    ax2 = ax1.twinx()
    l2, = ax2.plot(it_vals, [r['coverage'] for r in it_res], marker='s', color='r', label='cov', linewidth=2)
    ax2.set_ylabel('cov [0-1]', color='r'); ax2.tick_params(axis='y', labelcolor='r')
    ax1.legend(handles=[l1, l2], loc='center right'); ax1.set_title('Итерации')

    print(f"\n2. Параметр c (i=500, alpha=0.5)...")
    c_vals = sorted(set([0.01, 0.05, 0.1, c_rec, 0.3, 0.5, 1.0, 1.414, 2.0, 3.0]))
    c_res = evaluate_parameter(data, 'exploration_param', c_vals,
        {'iterations': 500, 'min_triplets': 5, 'coverage_threshold': cov_threshold, 'alpha': 0.5},
        max_idx, beta_a=beta_a, max_depth=max_depth, gamma_cov=gamma_cov,
        cov_threshold=cov_threshold, analysis_state=analysis_state, analysis_action=analysis_action)

    ax3 = plt.subplot(2, 3, 2)
    l3, = ax3.plot(c_vals, [r['ku'] for r in c_res], marker='o', color='b', label='ku', linewidth=2)
    l_cov, = ax3.plot(c_vals, [r['coverage'] for r in c_res], marker='^', color='r', label='cov', linewidth=2)
    ax3.set_xlabel('c (exploration)'); ax3.set_ylabel('ku / cov [0-1]'); ax3.set_xscale('log')
    if c_rec: ax3.axvline(x=c_rec, color='blue', linestyle='--', linewidth=1.5, label=f'c_rec={c_rec}')
    ax3.legend(); ax3.set_title('Параметр c (ku и cov)')

    print(f"\n3. Alpha (i=500, c={c_rec})...")
    a_vals = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    a_res = evaluate_parameter(data, 'alpha', a_vals,
        {'iterations': 500, 'min_triplets': 5, 'coverage_threshold': cov_threshold, 'exploration_param': c_rec},
        max_idx, beta_a=beta_a, max_depth=max_depth, gamma_cov=gamma_cov,
        cov_threshold=cov_threshold, analysis_state=analysis_state, analysis_action=analysis_action)

    ax5 = plt.subplot(2, 3, 3)
    l5, = ax5.plot(a_vals, [r['ku'] for r in a_res], marker='o', color='b', label='ku', linewidth=2)
    ax5.set_xlabel('alpha'); ax5.set_ylabel('ku', color='b'); ax5.tick_params(axis='y', labelcolor='b')
    ax6 = ax5.twinx()
    l6, = ax6.plot(a_vals, [r['norm_a'] for r in a_res], marker='s', color='r', label='norm_a', linewidth=2)
    ax6.set_ylabel('norm_a', color='r'); ax6.tick_params(axis='y', labelcolor='r')
    ax5.legend(handles=[l5, l6]); ax5.set_title('Alpha')

    # ku vs cov scatter
    ax7 = plt.subplot(2, 3, 4)
    all_ku = [r['ku'] for r in it_res + c_res + a_res]
    all_cov = [r['coverage'] for r in it_res + c_res + a_res]
    ax7.scatter(all_cov, all_ku, alpha=0.6, c='steelblue', edgecolors='navy')
    mv = max(max(all_ku, default=1), max(all_cov, default=1), 1)
    ax7.plot([0, mv], [0, mv], 'r--', alpha=0.5, label='ku = cov')
    ax7.set_xlabel('cov [0-1]'); ax7.set_ylabel('ku [0-1]')
    ax7.set_title('ku vs cov'); ax7.legend()

    # reward
    ax9 = plt.subplot(2, 3, 5)
    ax9.plot(c_vals, [r['reward'] for r in c_res], marker='o', color='green', label='reward (c)', linewidth=2)
    ax9.set_xlabel('c'); ax9.set_ylabel('reward'); ax9.set_xscale('log')
    ax9.legend(); ax9.set_title('Reward от c')

    # время
    ax11 = plt.subplot(2, 3, 6)
    ax11.plot(it_vals, [r['time'] for r in it_res], marker='o', color='teal', label='time (iter)', linewidth=2)
    ax11.plot(c_vals, [r['time'] for r in c_res], marker='s', color='olive', label='time (c)', linewidth=2)
    ax11.set_xlabel('Параметр'); ax11.set_ylabel('Время (с)'); ax11.legend(); ax11.set_title('Время')

    plt.tight_layout()
    plt.savefig(_results_path('mcts_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nГрафики сохранены: {_results_path('mcts_analysis.png')}")

# =====================================================================
# 7. ФИНАЛЬНЫЙ ПОИСК
# =====================================================================

def run_final_search(data, max_idx, auto=True, iterations=500,
                     alpha=0.5, exploration_param=1.414,
                     beta_a=beta_a_default, max_depth=None,
                     cov_thresh=None, gamma_cov=None, analysis=None,
                     init_state_idx=None, init_action_idx=None,
                     max_total_cols=None):
    print("\n" + "=" * 70)
    print("ФИНАЛЬНЫЙ ПОИСК ГИПОТЕЗЫ")
    print("=" * 70)
    if auto:
        c_tuned, max_depth_tuned, analysis, cov_thresh, gamma_cov = auto_tune(
            data, max_idx, iterations, alpha=alpha, beta_a=beta_a)
        exploration_param = c_tuned
        if max_depth is None: max_depth = max_depth_tuned
        print(f"\nИспользуем: c={exploration_param}, max_depth={max_depth}, "
              f"beta_a={beta_a}, cov_thresh={cov_thresh}, gamma_cov={gamma_cov}")
    else:
        if analysis is None:
            analysis = analyze_data_structure(data, max_idx)
        if cov_thresh is None:
            cov_thresh = 0.5
        if gamma_cov is None:
            gamma_cov = 3.0
        print(f"\nИспользуем: c={exploration_param}, max_depth={max_depth}, "
              f"beta_a={beta_a}, cov_thresh={cov_thresh}, gamma_cov={gamma_cov}")

    mcts = HypothesisMCTS(max_idx=max_idx, exploration_param=exploration_param,
                          beta_a=beta_a, max_depth=max_depth, gamma_cov=gamma_cov)
    start = time.time()
    analysis_state = analysis.get('state_cols', []) if analysis else []
    analysis_action = analysis.get('action_cols', []) if analysis else []

    # Если forced-индексы не входят в analysis — открываем все колонки
    if init_state_idx or init_action_idx:
        all_analysis = set(analysis_state + analysis_action)
        forced = set((init_state_idx or []) + (init_action_idx or []))
        if not forced.issubset(all_analysis):
            analysis_state = None
            analysis_action = None

    best_hyp, top_hyps = mcts.search(data, iterations=iterations, min_triplets=5, alpha=alpha,
                           coverage_threshold=cov_thresh,
                           init_state_idx=init_state_idx,
                           init_action_idx=init_action_idx,
                           analysis_state=analysis_state,
                           analysis_action=analysis_action,
                           max_total_cols=max_total_cols)
    elapsed = time.time() - start
    print(f"\nВремя: {elapsed:.2f} сек, Итераций: {iterations}")

    # Вывод топ-N (подробный)
    if top_hyps:
        print(f"\n--- Топ-{len(top_hyps)} гипотез ---")
        for i, h in enumerate(top_hyps):
            ku, norm_a, n_sas, n_a = h['ku']
            cov = h.get('coverage', 0.0)
            reward = h.get('reward', 0.0)
            depth = len(h['state_idx']) + len(h['action_idx'])
            print(f"  #{i+1}: S={h['state_idx']}, A={h['action_idx']} | "
                  f"ku={ku}, reward={reward:.4f}, norm_a={norm_a}, "
                  f"|sas'|={n_sas}, |a|={n_a}, cov={cov}, "
                  f"depth={depth}, time={elapsed:.2f}s")

    if best_hyp and best_hyp['state_idx'] and best_hyp['action_idx']:
        ku, norm_a, n_sas, n_a = best_hyp['ku']
        coverage = best_hyp['coverage']
        S = best_hyp['state_idx']; A = best_hyp['action_idx']
        na_cols = len(A) / max_idx if max_idx > 0 else 0
        reward = best_hyp.get('reward', ku - gamma_cov * coverage * coverage - alpha * norm_a - beta_a * na_cols)
        print(f"\nЛучшая гипотеза:")
        print(f"  State idx  : {S}")
        print(f"  Action idx : {A}")
        print(f"  ku         : {ku}")
        print(f"  norm_a     : {norm_a}")
        print(f"  |sas'|     : {n_sas}")
        print(f"  |a|        : {n_a}")
        print(f"  coverage   : {coverage}")
        print(f"  reward     : {round(reward, 4)}")
        print(f"  depth      : {len(S) + len(A)}")
        print(f"\nГраф (первые 20):")
        graph = build_graph_from_hypothesis(data, S, A)
        for i, (k, v) in enumerate(sorted(graph.items())):
            if i >= 20:
                print(f"  ... ещё {len(graph) - 20}"); break
            print(f"  {k}: {v}")
    else:
        print("\nГипотеза не найдена.")
    return best_hyp, top_hyps

# =====================================================================
# 8. MAIN
# =====================================================================

if __name__ == "__main__":
    # ====== ПРИНУДИТЕЛЬНЫЕ ИНДЕКСЫ (пустые = автопоиск) ======
    FORCED_STATE = [] #[3, 4, 6, 7]  
    FORCED_ACTION = [] #[2, 5]  

    ##data = load_real_data('data.csv')
    # ИСПОЛЬЗУЕМ ДИНАМИЧЕСКИЙ ПУТЬ ВМЕСТО 'data.csv'
    data = load_real_data(str(CSV_FILE_PATH))
    
    if not data:
        print("Сначала создайте data.csv через generate_data")
        exit(1)
    max_idx = len(data[0])
    print(f"max_idx = {max_idx}")

    # Вычисляем max_total_cols — ограничение общего числа колонок
    max_total_cols = None
    if FORCED_STATE or FORCED_ACTION:
        max_total_cols = len(FORCED_STATE) + len(FORCED_ACTION)

    if AUTO_TUNE:
        # ====== АВТОАНАЛИЗ ВКЛЮЧЁН ======
        c_rec, max_depth_rec, analysis, cov_thresh, gamma_cov = auto_tune(
            data, max_idx, DEFAULT_ITERATIONS, alpha=DEFAULT_ALPHA, beta_a=beta_a_default)

        print(f"\nРекомендация: c={c_rec}, max_depth={max_depth_rec}")
        print(f"Анализ: state={analysis['state_cols']}, action={analysis['action_cols']}, noise={analysis['noise_cols']}")

        # Sweep
        plot_results(data, max_idx, c_rec=c_rec, max_depth=max_depth_rec,
                     gamma_cov=gamma_cov, cov_threshold=cov_thresh,
                     analysis_state=analysis['state_cols'], analysis_action=analysis['action_cols'])

        # Финальный поиск
        best, top = run_final_search(data, max_idx, auto=False, iterations=DEFAULT_ITERATIONS,
                     alpha=DEFAULT_ALPHA, exploration_param=c_rec, max_depth=max_depth_rec,
                     beta_a=beta_a_default, cov_thresh=cov_thresh, gamma_cov=gamma_cov,
                     analysis=analysis,
                     init_state_idx=FORCED_STATE or None,
                     init_action_idx=FORCED_ACTION or None,
                     max_total_cols=max_total_cols)
    else:
        # ====== АВТОАНАЛИЗ ВЫКЛЮЧЁН ======
        print("\n" + "=" * 70)
        print("АВТОАНАЛИЗ ВЫКЛЮЧЁН. Ручные параметры:")
        print("=" * 70)
        print(f"  c={DEFAULT_C}, max_depth={DEFAULT_MAX_DEPTH}, cov_thresh={DEFAULT_COV_THRESH}, "
              f"gamma_cov={DEFAULT_GAMMA_COV}, iterations={DEFAULT_ITERATIONS}")
        print(f"  FORCED_STATE={FORCED_STATE}, FORCED_ACTION={FORCED_ACTION}")

        # Структурный анализ (без подробного отчёта, нужен для allowed_cols)
        analysis = analyze_data_structure(data, max_idx)
        expected_depth = analysis['expected_depth']
        c_rec = recommend_c_formula(max_idx, DEFAULT_ITERATIONS, expected_depth)
        B = 2 * max_idx
        D_reach = math.log(max(DEFAULT_ITERATIONS, 2)) / math.log(max(B, 2))

        print(f"\n  Анализ колонок:")
        print(f"    Всего колонок      : {max_idx}")
        print(f"    Фактор ветвления B : {B}")
        print(f"    State (оценка)     : {len(analysis['state_cols'])} {analysis['state_cols']}")
        print(f"    Action (оценка)    : {len(analysis['action_cols'])} {analysis['action_cols']}")
        print(f"    Noise (оценка)     : {len(analysis['noise_cols'])} {analysis['noise_cols']}")
        print(f"    Ожидаемая глубина  : {expected_depth}")
        print(f"    D_reachable (T={DEFAULT_ITERATIONS}) : {D_reach:.2f}")
        print(f"    c_recommended     : {c_rec} (не используется, c={DEFAULT_C})")
        print(f"\n    Детализация по колонкам:")
        for info in analysis['col_info']:
            print(f"      col {info['col']}: consist={info['consistency']}, autocorr={info['autocorr']} -> {info['class']}")
        print("=" * 70)

        # Финальный поиск с ручными параметрами
        best, top = run_final_search(data, max_idx, auto=False, iterations=DEFAULT_ITERATIONS,
                     alpha=DEFAULT_ALPHA, exploration_param=DEFAULT_C,
                     max_depth=DEFAULT_MAX_DEPTH,
                     beta_a=beta_a_default, cov_thresh=DEFAULT_COV_THRESH,
                     gamma_cov=DEFAULT_GAMMA_COV, analysis=analysis,
                     init_state_idx=FORCED_STATE or None,
                     init_action_idx=FORCED_ACTION or None,
                     max_total_cols=max_total_cols)

    # Запись результата в result.txt
    if best and best['state_idx'] and best['action_idx']:
        with open(_results_path('result.txt'), 'w', encoding='utf-8') as f:
            f.write(f"State idx  : {best['state_idx']}\n")
            f.write(f"Action idx : {best['action_idx']}\n")
            ku, norm_a, n_sas, n_a = best['ku']
            f.write(f"ku={ku}, norm_a={norm_a}, |sas'|={n_sas}, |a|={n_a}, "
                    f"cov={best.get('coverage', 0)}, reward={best.get('reward', 0):.4f}\n")
        print(f"\nРезультат записан в {_results_path('result.txt')}")


