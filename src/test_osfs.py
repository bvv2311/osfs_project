"""
Тестовый фреймворк для OSFS MCTS.

Схема:
  1. Генерируем синтетические данные с известным ground truth (state_idx, action_idx)
  2. Запускаем MCTS поиск
  3. Сравниваем найденную гипотезу с ground truth
  4. Считаем метрики: точность state, точность action, ku, reward

Запуск:
    python test_osfs.py              # все тесты
    python test_osfs.py --verbose    # подробный вывод
    python test_osfs.py --config=0   # только первый конфиг
"""

import csv
import sys
import time
import traceback
from pathlib import Path

# Добавляем src/ в путь
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from generate_csv_to_OSFS import generate_data, check_ku, _data_path
from OSFS_mcts import (
    HypothesisMCTS,
    run_final_search,
    analyze_data_structure,
    evaluate_hypothesis,
    beta_a_default,
)


# =====================================================================
#  МЕТРИКИ СРАВНЕНИЯ
# =====================================================================

def jaccard(set_a, set_b):
    """Коэффициент Жаккара: |A ∩ B| / |A ∪ B|"""
    sa, sb = set(set_a), set(set_b)
    if not sa and not sb:
        return 1.0
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


def precision_recall(true_set, pred_set):
    """Precision и Recall для множеств."""
    true_set, pred_set = set(true_set), set(pred_set)
    if not pred_set:
        return 0.0, 0.0
    tp = len(true_set & pred_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(true_set) if true_set else 0.0
    return precision, recall


def exact_match(true_state, true_action, pred_state, pred_action):
    """Точное совпадение гипотезы."""
    return (set(true_state) == set(pred_state) and
            set(true_action) == set(pred_action))


def state_action_overlap(true_state, true_action, pred_state, pred_action):
    """
    Проверяем, не попали ли true-state колонки в pred_action и наоборот.
    Возвращает (state_in_action, action_in_state) — сколько колонок перепутано.
    """
    ts, ta = set(true_state), set(true_action)
    ps, pa = set(pred_state), set(pred_action)
    state_in_action = len(ts & pa)
    action_in_state = len(ta & ps)
    return state_in_action, action_in_state


# =====================================================================
#  ОДИНОЧНЫЙ ТЕСТ
# =====================================================================

def run_single_test(config, mcts_params=None, verbose=False):
    """
    Запускает один тест: генерация -> поиск -> сравнение.

    config: dict параметров для generate_data
    mcts_params: dict параметров для MCTS (iterations, alpha, c, ...)
    verbose: печатать детали

    Возвращает dict с результатами.
    """
    if mcts_params is None:
        mcts_params = {}

    result = {
        'config': config.copy(),
        'ground_truth': None,
        'found': None,
        'exact_match': False,
        'state_jaccard': 0.0,
        'action_jaccard': 0.0,
        'state_precision': 0.0,
        'state_recall': 0.0,
        'action_precision': 0.0,
        'action_recall': 0.0,
        'state_in_action': 0,
        'action_in_state': 0,
        'ku_found': 0.0,
        'reward_found': 0.0,
        'ku_truth': None,
        'time_search': 0.0,
        'time_total': 0.0,
        'error': None,
    }

    t0 = time.time()

    try:
        # --- 1. Генерация данных ---
        filename = config.get('filename', f'test_{config.get("seed", 42)}.csv')
        config_copy = config.copy()
        config_copy['filename'] = filename

        gt = generate_data(**config_copy)
        result['ground_truth'] = {
            'state_idx': gt['state_idx'],
            'action_idx': gt['action_idx'],
            'noise_idx': gt['noise_idx'],
        }

        # --- 2. Проверка ku для ground truth ---
        ku_truth, skipped = check_ku(filename, gt['state_idx'], gt['action_idx'])
        result['ku_truth'] = ku_truth

        if verbose:
            print(f"  Ground truth: S={gt['state_idx']}, A={gt['action_idx']}, N={gt['noise_idx']}")
            print(f"  ku(truth) = {ku_truth}")

        # --- 3. Загружаем данные для MCTS ---
        data = []
        filepath = _data_path(filename)
        with open(filepath, 'r', newline='', encoding='utf-8') as f:
            for row in csv.reader(f):
                if row:
                    data.append(row)

        N = config['N']
        max_idx = N

        # --- 4. Автоанализ ---
        analysis = analyze_data_structure(data, max_idx)

        # --- 5. MCTS поиск ---
        iterations = mcts_params.get('iterations', 500)
        alpha = mcts_params.get('alpha', 0.5)
        exploration_param = mcts_params.get('exploration_param', 1.414)
        coverage_threshold = mcts_params.get('coverage_threshold', 0.95)
        gamma_cov = mcts_params.get('gamma_cov', 3.0)
        max_depth = mcts_params.get('max_depth', None)
        beta_a = mcts_params.get('beta_a', beta_a_default)

        mcts = HypothesisMCTS(
            max_idx=max_idx,
            exploration_param=exploration_param,
            beta_a=beta_a,
            max_depth=max_depth,
            gamma_cov=gamma_cov,
        )

        t_search = time.time()
        best_hyp, top_hyps = mcts.search(
            data,
            iterations=iterations,
            min_triplets=5,
            alpha=alpha,
            coverage_threshold=coverage_threshold,
            analysis_state=analysis['state_cols'],
            analysis_action=analysis['action_cols'],
        )
        result['time_search'] = time.time() - t_search

        # --- 6. Сравнение ---
        if best_hyp and best_hyp['state_idx'] and best_hyp['action_idx']:
            pred_s = best_hyp['state_idx']
            pred_a = best_hyp['action_idx']
            result['found'] = {
                'state_idx': pred_s,
                'action_idx': pred_a,
                'ku': best_hyp['ku'],
                'coverage': best_hyp.get('coverage', 0.0),
                'reward': best_hyp.get('reward', 0.0),
            }
            result['ku_found'] = best_hyp['ku'][0]
            result['reward_found'] = best_hyp.get('reward', 0.0)

            result['exact_match'] = exact_match(
                gt['state_idx'], gt['action_idx'], pred_s, pred_a)

            result['state_jaccard'] = jaccard(gt['state_idx'], pred_s)
            result['action_jaccard'] = jaccard(gt['action_idx'], pred_a)

            sp, sr = precision_recall(gt['state_idx'], pred_s)
            ap, ar = precision_recall(gt['action_idx'], pred_a)
            result['state_precision'] = sp
            result['state_recall'] = sr
            result['action_precision'] = ap
            result['action_recall'] = ar

            sia, ais = state_action_overlap(
                gt['state_idx'], gt['action_idx'], pred_s, pred_a)
            result['state_in_action'] = sia
            result['action_in_state'] = ais

            if verbose:
                print(f"  Found:      S={pred_s}, A={pred_a}")
                print(f"  ku(found) = {best_hyp['ku']}")
                print(f"  Exact match: {result['exact_match']}")
                print(f"  State  Jaccard={result['state_jaccard']:.3f}  "
                      f"P={sp:.3f} R={sr:.3f}")
                print(f"  Action Jaccard={result['action_jaccard']:.3f}  "
                      f"P={ap:.3f} R={ar:.3f}")
                if sia or ais:
                    print(f"  !! Перепутаны: state->action={sia}, action->state={ais}")

                if top_hyps:
                    print(f"  Топ-{len(top_hyps)}:")
                    for i, h in enumerate(top_hyps):
                        print(f"    #{i+1}: S={h['state_idx']}, A={h['action_idx']}, "
                              f"ku={h['ku'][0]}, reward={h.get('reward',0):.4f}")
        else:
            if verbose:
                print("  !! Гипотеза не найдена!")

    except Exception as e:
        result['error'] = str(e)
        if verbose:
            traceback.print_exc()

    result['time_total'] = time.time() - t0
    return result


# =====================================================================
#  НАБОР ТЕСТОВЫХ КОНФИГУРАЦИЙ
# =====================================================================

TEST_CONFIGS = [
    # --- 0: Базовый (4 state, 2 action, 2 noise) ---
    {
        'name': 'base_4s2a2n',
        'num_rows': 5000, 'N': 8,
        'n_state_cols': 4, 'n_action_cols': 2,
        'state_ranges': [2, 3, 5, 4],
        'action_ranges': [3, 4],
        'slip_prob': 0.002, 'noise_max': 4,
        'seed': 42, 'missing_prob': 0.005, 'empty_row_prob': 0.002,
    },
    # --- 1: Меньше данных ---
    {
        'name': 'small_500rows',
        'num_rows': 500, 'N': 8,
        'n_state_cols': 4, 'n_action_cols': 2,
        'state_ranges': [2, 3, 5, 4],
        'action_ranges': [3, 4],
        'slip_prob': 0.002, 'noise_max': 4,
        'seed': 42, 'missing_prob': 0.005, 'empty_row_prob': 0.002,
    },
    # --- 2: Больше шума (slip) ---
    {
        'name': 'high_slip_05',
        'num_rows': 5000, 'N': 8,
        'n_state_cols': 4, 'n_action_cols': 2,
        'state_ranges': [2, 3, 5, 4],
        'action_ranges': [3, 4],
        'slip_prob': 0.05, 'noise_max': 4,
        'seed': 42, 'missing_prob': 0.005, 'empty_row_prob': 0.002,
    },
    # --- 3: Много пропусков ---
    {
        'name': 'missing_10pct',
        'num_rows': 5000, 'N': 8,
        'n_state_cols': 4, 'n_action_cols': 2,
        'state_ranges': [2, 3, 5, 4],
        'action_ranges': [3, 4],
        'slip_prob': 0.002, 'noise_max': 4,
        'seed': 42, 'missing_prob': 0.10, 'empty_row_prob': 0.05,
    },
    # --- 4: Простой граф (2 state, 1 action) ---
    {
        'name': 'simple_2s1a',
        'num_rows': 3000, 'N': 6,
        'n_state_cols': 2, 'n_action_cols': 1,
        'state_ranges': [5, 5],
        'action_ranges': [4],
        'slip_prob': 0.0, 'noise_max': 3,
        'seed': 7, 'missing_prob': 0.0, 'empty_row_prob': 0.0,
    },
    # --- 5: Без шума, чистый граф ---
    {
        'name': 'no_noise_cols',
        'num_rows': 5000, 'N': 6,
        'n_state_cols': 4, 'n_action_cols': 2,
        'state_ranges': [2, 3, 5, 4],
        'action_ranges': [3, 4],
        'slip_prob': 0.0, 'noise_max': 0,
        'seed': 42, 'missing_prob': 0.0, 'empty_row_prob': 0.0,
    },
    # --- 6: Большой граф (6 state, 3 action, 3 noise) ---
    {
        'name': 'large_6s3a3n',
        'num_rows': 8000, 'N': 12,
        'n_state_cols': 6, 'n_action_cols': 3,
        'state_ranges': [2, 2, 3, 2, 2, 3],
        'action_ranges': [2, 3, 2],
        'slip_prob': 0.005, 'noise_max': 5,
        'seed': 99, 'missing_prob': 0.005, 'empty_row_prob': 0.002,
    },
    # --- 7: Разные seed (воспроизводимость) ---
    {
        'name': 'seed_123',
        'num_rows': 5000, 'N': 8,
        'n_state_cols': 4, 'n_action_cols': 2,
        'state_ranges': [2, 3, 5, 4],
        'action_ranges': [3, 4],
        'slip_prob': 0.002, 'noise_max': 4,
        'seed': 123, 'missing_prob': 0.005, 'empty_row_prob': 0.002,
    },
]

# Параметры MCTS для тестов
MCTS_PARAMS = {
    'iterations': 500,
    'alpha': 0.5,
    'exploration_param': 1.414,
    'coverage_threshold': 0.95,
    'gamma_cov': 3.0,
}


# =====================================================================
#  ЗАПУСК ВСЕХ ТЕСТОВ
# =====================================================================

def run_all_tests(verbose=False, config_idx=None):
    """Запускает все тесты и печатает сводку."""
    configs = list(TEST_CONFIGS)
    if config_idx is not None:
        configs = [TEST_CONFIGS[config_idx]]

    results = []
    print("=" * 90)
    print("ТЕСТИРОВАНИЕ OSFS MCTS")
    print("=" * 90)
    print(f"Параметры MCTS: {MCTS_PARAMS}")
    print()

    for i, cfg in enumerate(configs):
        name = cfg.pop('name', f'test_{i}')
        print(f"--- Тест {i}: {name} ---")
        print(f"  N={cfg['N']}, state={cfg['n_state_cols']}, action={cfg['n_action_cols']}, "
              f"rows={cfg['num_rows']}, slip={cfg['slip_prob']}, "
              f"missing={cfg['missing_prob']}")

        res = run_single_test(cfg, MCTS_PARAMS, verbose=verbose)
        res['name'] = name
        results.append(res)

        if res['error']:
            print(f"  ERROR: {res['error']}")
        elif not res['found']:
            print(f"  WARNING: Гипотеза не найдена")
        else:
            status = "OK" if res['exact_match'] else "PARTIAL"
            print(f"  [{status}] Exact={res['exact_match']}  "
                  f"S_jac={res['state_jaccard']:.3f}  A_jac={res['action_jaccard']:.3f}  "
                  f"ku={res['ku_found']:.3f}  reward={res['reward_found']:.4f}  "
                  f"t={res['time_search']:.1f}s")
        print()

    # --- Сводная таблица ---
    print("=" * 90)
    print("СВОДНАЯ ТАБЛИЦА")
    print("=" * 90)
    header = (f"{'#':>2} {'Name':<18} {'S_jac':>6} {'A_jac':>6} "
              f"{'S_P':>5} {'S_R':>5} {'A_P':>5} {'A_R':>5} "
              f"{'ku':>6} {'rew':>7} {'ex':>3} {'time':>6}")
    print(header)
    print("-" * 90)

    n_pass = 0
    n_partial = 0
    n_fail = 0

    for i, r in enumerate(results):
        if r['error'] or not r['found']:
            print(f"{i:>2} {r.get('name','?'):<18} {'ERR':>6} {'ERR':>6} "
                  f"{'-':>5} {'-':>5} {'-':>5} {'-':>5} "
                  f"{'-':>6} {'-':>7} {'-':>3} {'-':>6}")
            n_fail += 1
            continue

        em = "Y" if r['exact_match'] else "N"
        if r['exact_match']:
            n_pass += 1
        else:
            n_partial += 1

        print(f"{i:>2} {r['name']:<18} "
              f"{r['state_jaccard']:>6.3f} {r['action_jaccard']:>6.3f} "
              f"{r['state_precision']:>5.2f} {r['state_recall']:>5.2f} "
              f"{r['action_precision']:>5.2f} {r['action_recall']:>5.2f} "
              f"{r['ku_found']:>6.3f} {r['reward_found']:>7.4f} "
              f"{em:>3} {r['time_search']:>5.1f}s")

    print("-" * 90)
    total = len(results)
    print(f"Точно: {n_pass}/{total}  Частично: {n_partial}/{total}  "
          f"Ошибка: {n_fail}/{total}")
    print()

    # --- Детали по перепутанным колонкам ---
    swapped = [r for r in results if r.get('state_in_action', 0) > 0 or r.get('action_in_state', 0) > 0]
    if swapped:
        print("ПЕРЕПУТАННЫЕ КОЛОНКИ (state <-> action):")
        for r in swapped:
            print(f"  {r['name']}: state->action={r['state_in_action']}, "
                  f"action->state={r['action_in_state']}")
        print()

    # --- ku: ground truth vs found ---
    print("СРАВНЕНИЕ ku (ground truth vs found):")
    for r in results:
        if r.get('ku_truth') and r.get('found'):
            ku_t = r['ku_truth'][0] if isinstance(r['ku_truth'], tuple) else r['ku_truth']
            print(f"  {r['name']}: ku_truth={ku_t}, ku_found={r['ku_found']}, "
                  f"delta={ku_t - r['ku_found']:+.3f}")
    print()

    return results


# =====================================================================
#  MAIN
# =====================================================================

if __name__ == '__main__':
    verbose = '--verbose' in sys.argv or '-v' in sys.argv
    config_idx = None

    for arg in sys.argv[1:]:
        if arg.startswith('--config='):
            config_idx = int(arg.split('=')[1])
        elif arg.startswith('--cfg='):
            config_idx = int(arg.split('=')[1])

    results = run_all_tests(verbose=verbose, config_idx=config_idx)
