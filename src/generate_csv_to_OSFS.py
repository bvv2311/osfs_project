import csv
import random
import itertools
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Автоопределение пути к /data (скрипт лежит в /src, данные — в /data)
_SRC_DIR = Path(__file__).resolve().parent
_DATA_DIR = _SRC_DIR.parent / 'data'
_DATA_DIR.mkdir(parents=True, exist_ok=True)

def _data_path(filename):
    """Присоединяет путь к /data, если filename — просто имя файла."""
    p = Path(filename)
    if p.is_absolute() or len(p.parts) > 1:
        return str(p)
    return str(_DATA_DIR / filename)


# =====================================================================
#  ГЕНЕРАТОР N-МЕРНОГО ГРАФА ПЕРЕХОДОВ
#  Единый формат: задаёте config — получаете CSV + ground truth
#  Поддержка частичной наблюдаемости: пустые ячейки и пустые строки
# =====================================================================

def generate_data(
    filename='data.csv',
    num_rows=1000,
    N=8,
    n_state_cols=2,
    n_action_cols=1,
    state_ranges=None,
    action_ranges=None,
    slip_prob=0.01,
    noise_max=9,
    seed=42,
    missing_prob=0.05,
    empty_row_prob=0.02,
):
    """
    Генератор n-мерного графа переходов с произвольным числом действий
    и произвольным диапазоном значений в state и action колонках.

    Параметры:
        N              — всего столбцов в CSV
        n_state_cols   — число state-колонок (размерность пространства)
        n_action_cols  — число action-колонок
        state_ranges   — размер решётки по каждой оси [0..r-1]
        action_ranges  — число значений в каждой action-колонке
        slip_prob      — вероятность проскальзывания (неоднозначность)
        noise_max      — макс значение шумовых колонок (0..noise_max)
        seed           — seed для воспроизводимости
        missing_prob   — вероятность пропуска отдельной ячейки (0..1)
        empty_row_prob — вероятность целиком пустой строки (0..1)

    Возвращает dict с истинной структурой.
    """
    random.seed(seed)

    if state_ranges is None:
        state_ranges = [10] * n_state_cols
    if action_ranges is None:
        action_ranges = [4] * n_action_cols

    assert len(state_ranges) == n_state_cols
    assert len(action_ranges) == n_action_cols

    n_noise = N - n_state_cols - n_action_cols
    assert n_noise >= 0
    assert all(r >= 2 for r in state_ranges)
    assert all(r >= 2 for r in action_ranges)

    total_actions = 1
    for r in action_ranges:
        total_actions *= r
    move_vectors = _build_move_vectors(n_state_cols, total_actions)

    max_unique = 3 ** n_state_cols
    if total_actions > max_unique:
        n_dup = total_actions - max_unique
        print(f"  ВНИМАНИЕ: total_actions={total_actions} > 3^{n_state_cols}={max_unique}")
        print(f"  {n_dup} действий имеют дублирующие векторы перемещения")
        print(f"  ku будет ниже из-за неоднозначности даже при slip_prob=0\n")

    all_indices = list(range(N))
    random.shuffle(all_indices)
    state_idx = sorted(all_indices[:n_state_cols])
    action_idx = sorted(all_indices[n_state_cols:n_state_cols + n_action_cols])
    noise_idx = sorted(all_indices[n_state_cols + n_action_cols:])

    pos = [r // 2 for r in state_ranges]

    rows = []
    n_empty_cells = 0
    n_empty_rows = 0

    for _ in range(num_rows):
        # --- Целиком пустая строка ---
        if random.random() < empty_row_prob:
            rows.append([''] * N)
            n_empty_rows += 1
            continue

        noise_vals = [random.randint(0, noise_max) for _ in noise_idx]
        action_idx_val = random.randint(0, total_actions - 1)
        action_vals = _decode_action(action_idx_val, action_ranges)

        if random.random() < slip_prob:
            actual_idx = random.randint(0, total_actions - 1)
        else:
            actual_idx = action_idx_val

        move = move_vectors[actual_idx]

        row = [None] * N
        for k, si in enumerate(state_idx):
            row[si] = pos[k]
        for k, ai in enumerate(action_idx):
            row[ai] = action_vals[k]
        for k, ni in enumerate(noise_idx):
            row[ni] = noise_vals[k]

        # --- Пропуск отдельных ячеек ---
        out_row = []
        for v in row:
            if v is not None and random.random() < missing_prob:
                out_row.append('')
                n_empty_cells += 1
            else:
                out_row.append(str(v) if v is not None else '')
        rows.append(out_row)

        for d in range(n_state_cols):
            pos[d] = max(0, min(state_ranges[d] - 1, pos[d] + move[d]))

    filepath = _data_path(filename)
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)

    print(f"Файл '{filepath}' создан ({num_rows} строк)")
    print(f"  N={N}, n_state={n_state_cols}, n_action={n_action_cols}, noise={n_noise}")
    print(f"  state_idx  = {state_idx}  (ranges={state_ranges})")
    print(f"  action_idx = {action_idx} (ranges={action_ranges})")
    print(f"  noise_idx  = {noise_idx}  (0..{noise_max})")
    print(f"  total_actions = {total_actions}, slip_prob={slip_prob}")
    print(f"  missing_prob={missing_prob}, empty_row_prob={empty_row_prob}")
    print(f"  пропущено ячеек: {n_empty_cells}, пустых строк: {n_empty_rows}")

    return {
        'state_idx': state_idx,
        'action_idx': action_idx,
        'noise_idx': noise_idx,
        'state_ranges': state_ranges,
        'action_ranges': action_ranges,
        'total_actions': total_actions,
        'move_vectors': move_vectors,
    }


def _build_move_vectors(n_dims, total_actions):
    vectors = []
    used = set()
    for d in range(n_dims):
        v = [0] * n_dims; v[d] = 1
        vectors.append(v); used.add(tuple(v))
        v = [0] * n_dims; v[d] = -1
        vectors.append(v); used.add(tuple(v))
    vectors.append([0] * n_dims); used.add(tuple([0] * n_dims))
    for combo in itertools.product([-1, 0, 1], repeat=n_dims):
        if len(vectors) >= total_actions:
            break
        if combo not in used:
            vectors.append(list(combo))
            used.add(combo)
    if len(vectors) < total_actions:
        base = list(vectors)
        i = 0
        while len(vectors) < total_actions:
            vectors.append(list(base[i % len(base)]))
            i += 1
    return vectors[:total_actions]


def _decode_action(action_idx, action_ranges):
    vals = []
    idx = action_idx
    for r in reversed(action_ranges):
        vals.append(idx % r)
        idx //= r
    return list(reversed(vals))


# =====================================================================
#  ФУНКЦИИ ПРОВЕРКИ ku
# =====================================================================

def build_obj(triplets):
    graph = {}
    for state, action, next_state in triplets:
        if state not in graph:
            graph[state] = [[[], action]]
        actions_list = [item[1] for item in graph[state]]
        if action not in actions_list:
            graph[state].append([[next_state], action])
        else:
            i = actions_list.index(action)
            if next_state not in graph[state][i][0]:
                graph[state][i][0].append(next_state)
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
        return (round(1 - u/(n+u), 3),
                round(len(set(actions))/(n+u), 3),
                n+u, len(set(actions)))
    return None


def _is_empty_row(row):
    """Проверка: строка целиком пустая (все значения пустые)."""
    return all(str(v).strip() == '' for v in row)


def _has_empty_cell(row, cols):
    """Проверка: есть ли пустая ячейка среди указанных колонок."""
    for c in cols:
        if c >= len(row) or str(row[c]).strip() == '':
            return True
    return False


def check_ku(filename, state_idx, action_idx):
    """Проверяет ku для заданной гипотезы на CSV-файле.
    Пропускает строки с пустыми ячейками в нужных колонках."""
    data = []
    filepath = _data_path(filename)
    with open(filepath, 'r', newline='') as f:
        for row in csv.reader(f):
            if row:
                data.append(row)
    triplets = []
    skipped = 0
    for i in range(len(data) - 1):
        cur, nxt = data[i], data[i + 1]
        needed = list(state_idx) + list(action_idx)
        if _is_empty_row(cur) or _is_empty_row(nxt):
            skipped += 1
            continue
        if _has_empty_cell(cur, needed) or _has_empty_cell(nxt, state_idx):
            skipped += 1
            continue
        try:
            s = tuple(int(cur[j]) for j in state_idx)
            a = tuple(int(cur[k]) for k in action_idx)
            ns = tuple(int(nxt[l]) for l in state_idx)
            triplets.append((s, a, ns))
        except (ValueError, IndexError):
            skipped += 1
            continue
    g = build_obj(triplets)
    result = ku_get(g)
    return result, skipped


def show_obj(graph):
    for k, v in sorted(graph.items()):
        print(f"{k}: {v}")


# =====================================================================
#  ЗАПУСК — меняйте только CONFIG
# =====================================================================

if __name__ == '__main__':
    CONFIG = {
        'filename':       'data.csv',
        'num_rows':        5000,
        'N':              8,
        'n_state_cols':    4,
        'n_action_cols':   2,
        'state_ranges':   [2,3, 5,4],
        'action_ranges':  [3, 4],
        'slip_prob':      0.05,
        'noise_max':       4,
        'seed':           42,
        'missing_prob':    0.005,    # ячеек пустые
        'empty_row_prob':  0.002,    # строк целиком пустые
    }

    print("=" * 70)
    print("ГЕНЕРАЦИЯ ДАННЫХ")
    print("=" * 70)
    info = generate_data(**CONFIG)

    print("\n" + "=" * 70)
    print("ПРОВЕРКА: ku для истинной гипотезы")
    print("=" * 70)
    ku, skipped = check_ku(CONFIG['filename'], info['state_idx'], info['action_idx'])
    print(f"  state_idx  = {info['state_idx']}")
    print(f"  action_idx = {info['action_idx']}")
    print(f"  ku = {ku}")
    print(f"  пропущено строк (пустые ячейки): {skipped}")

    if info['noise_idx']:
        print("\nПроверка: шумовые колонки не должны давать высокий ku")
        ku_noise, _ = check_ku(CONFIG['filename'], info['noise_idx'][:2],
                               info['noise_idx'][2:3] if len(info['noise_idx']) > 2 else info['noise_idx'][:1])
        print(f"  noise_idx  = {info['noise_idx']}")
        print(f"  ku(noise)  = {ku_noise}")

    total_actions = 1
    for r in CONFIG['action_ranges']:
        total_actions *= r
    max_unique = 3 ** CONFIG['n_state_cols']
    print(f"\n  total_actions = {total_actions}")
    print(f"  3^n_state    = {max_unique}")
    if total_actions <= max_unique:
        print(f"  Все {total_actions} векторов перемещений уникальны")
    else:
        print(f"  Внимание: {total_actions - max_unique} дубликатов (ku будет ниже)")

    print("\nГотово.")

 
