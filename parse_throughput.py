#!/usr/bin/env python3
"""parse_throughput.py - Solo Throughput 실험 후처리 스크립트

서버에서 데이터를 scp로 가져온 후 로컬(Mac)에서 실행.

기능:
  1. solo_timestamps.csv에서 throughput 확인/재파싱
  2. power CSV에서 워크로드 구간(elapsed 30~150s) 평균 전력 계산
  3. wall power CSV 매칭 (raspi 측정, 타임스탬프 기반)
  4. 에너지 효율 계산: efficiency = throughput / avg_power_W
  5. throughput_summary.csv 출력

사용법:
  python3 parse_throughput.py --data-dir ~/urop/data/20260601_2300/
  python3 parse_throughput.py --data-dir ~/urop/data/20260601_2300/ --wall-csv wall_throughput_0601.csv
  python3 parse_throughput.py --data-dir ~/urop/data/20260601_2300/ --reparse
"""

import argparse
import csv
import os
import re
import sys
from datetime import datetime
from pathlib import Path


# ==========================================
# Power CSV 파싱
# ==========================================
def parse_power_csv(csv_path, start_elapsed=30, end_elapsed=150):
    """concurrent_power.py 출력 CSV에서 워크로드 구간의 평균 전력 추출.

    Args:
        csv_path: power CSV 파일 경로
        start_elapsed: 워크로드 시작 시점 (초)
        end_elapsed: 워크로드 종료 시점 (초)

    Returns:
        dict with average power values, or None on failure.
    """
    if not os.path.exists(csv_path):
        return None

    rows = []
    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    elapsed = float(row.get("elapsed_s", 0))
                except (ValueError, TypeError):
                    continue
                if start_elapsed <= elapsed <= end_elapsed:
                    rows.append(row)
    except Exception as e:
        print(f"  WARNING: power CSV 읽기 실패: {csv_path}: {e}")
        return None

    if not rows:
        print(f"  WARNING: 워크로드 구간 데이터 없음: {csv_path}")
        return None

    # 평균 전력 계산
    def avg_col(col_name):
        vals = []
        for row in rows:
            try:
                v = float(row.get(col_name, 0))
                vals.append(v)
            except (ValueError, TypeError):
                pass
        return sum(vals) / len(vals) if vals else 0.0

    # RAPL 컬럼 이름 탐색 (서버에 따라 다를 수 있음)
    sample_keys = rows[0].keys() if rows else []

    # RAPL package (CPU) 전력
    rapl_pkg_col = None
    for key in sample_keys:
        if "rapl" in key.lower() and "package" in key.lower():
            rapl_pkg_col = key
            break
    if rapl_pkg_col is None:
        for key in sample_keys:
            if "rapl" in key.lower() and "pkg" in key.lower():
                rapl_pkg_col = key
                break

    # RAPL DRAM 전력
    rapl_dram_col = None
    for key in sample_keys:
        if "rapl" in key.lower() and "dram" in key.lower():
            rapl_dram_col = key
            break

    # GPU 전력
    gpu0_col = None
    gpu1_col = None
    for key in sample_keys:
        if key == "gpu0_W":
            gpu0_col = key
        elif key == "gpu1_W":
            gpu1_col = key

    result = {
        "rapl_pkg_W": avg_col(rapl_pkg_col) if rapl_pkg_col else 0.0,
        "rapl_dram_W": avg_col(rapl_dram_col) if rapl_dram_col else 0.0,
        "gpu0_W": avg_col(gpu0_col) if gpu0_col else 0.0,
        "gpu1_W": avg_col(gpu1_col) if gpu1_col else 0.0,
        "cpu_util_pct": avg_col("cpu_util_pct"),
        "n_samples": len(rows),
    }

    # 시스템 총 전력 (RAPL + GPU) - wall power는 별도 측정 필요
    result["system_W"] = (result["rapl_pkg_W"] + result["rapl_dram_W"]
                          + result["gpu0_W"] + result["gpu1_W"])

    return result


# ==========================================
# Wall Power CSV 매칭
# ==========================================
def load_wall_power(wall_csv_path):
    """wall_power.py 출력 CSV를 로드.

    Returns:
        list of (unix_timestamp, wall_W) tuples, sorted by time.
    """
    if not wall_csv_path or not os.path.exists(wall_csv_path):
        return None

    data = []
    try:
        with open(wall_csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ts = float(row["timestamp"])
                    wall_w = float(row["wall_W"])
                    data.append((ts, wall_w))
                except (ValueError, KeyError):
                    continue
    except Exception as e:
        print(f"WARNING: wall CSV 읽기 실패: {wall_csv_path}: {e}")
        return None

    if not data:
        return None

    data.sort(key=lambda x: x[0])
    print(f"wall power: {len(data)}개 샘플 로드됨 "
          f"({data[0][0]:.0f} ~ {data[-1][0]:.0f})")
    return data


def match_wall_power(wall_data, start_time_str, idle_wait=30, workload_dur=120):
    """실험 시작 시간 기준으로 워크로드 구간의 wall power 평균 계산.

    Args:
        wall_data: load_wall_power() 결과
        start_time_str: "YYYY-MM-DD HH:MM:SS" 형식
        idle_wait: 워크로드 시작까지의 대기 시간 (초)
        workload_dur: 워크로드 실행 시간 (초)

    Returns:
        average wall_W during workload period, or None.
    """
    if not wall_data or not start_time_str:
        return None

    try:
        dt = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
        # 서버 시간 기준 Unix timestamp
        start_epoch = dt.timestamp()
    except (ValueError, TypeError):
        return None

    # 워크로드 구간: start + idle_wait ~ start + idle_wait + workload_dur
    wl_start = start_epoch + idle_wait
    wl_end = start_epoch + idle_wait + workload_dur

    # 해당 구간의 wall power 샘플 추출
    values = [w for (ts, w) in wall_data if wl_start <= ts <= wl_end]

    if not values:
        # 시간대 불일치 가능 (서버 vs raspi 시계 차이)
        # ±60초 여유를 두고 재시도
        values = [w for (ts, w) in wall_data
                  if (wl_start - 60) <= ts <= (wl_end + 60)]
        if values:
            print(f"  NOTE: wall power ±60s 보정 적용 (samples={len(values)})")

    if not values:
        return None

    return sum(values) / len(values)


# ==========================================
# Throughput 재파싱 (stdout 로그에서)
# ==========================================
def _read_stderr_fallback(stdout_path):
    """stdout 경로에서 대응하는 stderr 경로를 추론하여 읽기."""
    stderr_path = stdout_path.replace("/stdout/", "/stderr/").replace(".log", ".err")
    try:
        with open(stderr_path, "r") as f:
            return f.read()
    except FileNotFoundError:
        return None


def reparse_throughput(stdout_path, workload_name):
    """stdout 로그에서 throughput 재파싱.

    run_solo_throughput.py와 동일한 로직.
    """
    try:
        with open(stdout_path, "r") as f:
            content = f.read()
    except FileNotFoundError:
        content = ""

    # nodejs/autocannon: stdout이 비어있어도 stderr에 데이터 있을 수 있음
    if workload_name == "nodejs":
        for text in [content, _read_stderr_fallback(stdout_path)]:
            if not text:
                continue
            text = re.sub(r'\x1b\[[0-9;]*m', '', text)
            for line in text.split('\n'):
                if 'Req/Sec' in line and '\u2502' in line:
                    parts = [p.strip() for p in line.split('\u2502') if p.strip()]
                    if len(parts) >= 6:
                        try:
                            return float(parts[5])
                        except ValueError:
                            pass
            match = re.search(r'"average"\s*:\s*(\d+\.?\d*)', text)
            if match:
                return float(match.group(1))
        return None

    if not content.strip():
        return None

    # ffmpeg: -progress pipe:1 output
    if workload_name == "ffmpeg":
        fps_values = re.findall(r"^fps=(\d+\.?\d*)$", content, re.MULTILINE)
        if not fps_values:
            return None
        valid_fps = [float(v) for v in fps_values if float(v) > 0]
        if not valid_fps:
            return None
        warmup_cutoff = max(1, len(valid_fps) // 5)
        steady_fps = valid_fps[warmup_cutoff:]
        if not steady_fps:
            steady_fps = valid_fps
        return sum(steady_fps) / len(steady_fps)

    # 일반 워크로드: "Done." 라인
    match = re.search(r"Done\..*\((\d+\.?\d*)\s+\w+/s\)", content)
    if match:
        return float(match.group(1))

    match = re.search(r"Done\..*\((\d+\.?\d*)\s+fps\)", content)
    if match:
        return float(match.group(1))

    # llm (괄호 없는 형식): tokens / seconds
    match = re.search(
        r"Done\.\s+\d+\s+iterations,\s+(\d+)\s+tokens\s+in\s+(\d+\.?\d*)s",
        content
    )
    if match:
        tokens = int(match.group(1))
        seconds = float(match.group(2))
        if seconds > 0:
            return tokens / seconds

    return None


# ==========================================
# 메인
# ==========================================
def main():
    parser = argparse.ArgumentParser(
        description="Solo Throughput 실험 후처리: throughput + power → efficiency")
    parser.add_argument("--data-dir", required=True,
                        help="실험 데이터 디렉토리 (logs/YYYYMMDD_HHMM/)")
    parser.add_argument("--reparse", action="store_true",
                        help="stdout에서 throughput 재파싱")
    parser.add_argument("--power-start", type=float, default=30,
                        help="power 분석 시작 시점 (기본: 30s)")
    parser.add_argument("--power-end", type=float, default=150,
                        help="power 분석 종료 시점 (기본: 150s)")
    parser.add_argument("--wall-csv", default=None,
                        help="wall_power.py 출력 CSV 경로 (raspi 측정)")
    parser.add_argument("-o", "--output", default=None,
                        help="출력 CSV 경로 (기본: data-dir/throughput_summary.csv)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: 디렉토리 없음: {data_dir}")
        sys.exit(1)

    # solo_timestamps.csv 읽기
    ts_csv_path = data_dir / "solo_timestamps.csv"
    if not ts_csv_path.exists():
        print(f"ERROR: solo_timestamps.csv 없음: {ts_csv_path}")
        sys.exit(1)

    experiments = []
    with open(ts_csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            experiments.append(row)

    if not experiments:
        print("ERROR: 실험 데이터 없음")
        sys.exit(1)

    print(f"실험 {len(experiments)}개 로드됨")
    print(f"power 구간: {args.power_start}~{args.power_end}s")

    # wall power 로드
    wall_data = None
    if args.wall_csv:
        wall_data = load_wall_power(args.wall_csv)
        if wall_data is None:
            print(f"WARNING: wall CSV 로드 실패, wall_W 없이 진행")

    # 결과 수집
    results = []
    for exp in experiments:
        workload = exp["workload"]
        cores = exp["cores"]
        rep = exp["rep"]
        exp_tag = f"{workload}_{cores}c_r{rep}"

        # throughput
        throughput = None
        unit = exp.get("throughput_unit", "")

        if args.reparse:
            # stdout에서 재파싱
            stdout_path = data_dir / "stdout" / f"{exp_tag}.log"
            throughput = reparse_throughput(str(stdout_path), workload)
            if throughput is not None:
                print(f"  {exp_tag}: reparsed throughput = {throughput:.2f} {unit}")
        else:
            # CSV에서 기존 값 사용
            tp_str = exp.get("throughput", "")
            if tp_str:
                try:
                    throughput = float(tp_str)
                except ValueError:
                    pass

        # power CSV 파싱
        power_csv = data_dir / "power" / f"{exp_tag}.csv"
        power = parse_power_csv(str(power_csv), args.power_start, args.power_end)

        # wall power 매칭
        wall_w = None
        if wall_data:
            start_time_str = exp.get("start_time", "")
            wall_w = match_wall_power(wall_data, start_time_str)

        # 에너지 효율 계산 (wall power 우선, 없으면 RAPL+GPU 합산)
        efficiency = None
        if throughput:
            if wall_w and wall_w > 0:
                efficiency = throughput / wall_w
            elif power and power["system_W"] > 0:
                efficiency = throughput / power["system_W"]

        result = {
            "exp_id": exp.get("exp_id", ""),
            "workload": workload,
            "cores": cores,
            "rep": rep,
            "throughput": throughput,
            "throughput_unit": unit,
            "rapl_pkg_W": power["rapl_pkg_W"] if power else "",
            "rapl_dram_W": power["rapl_dram_W"] if power else "",
            "gpu0_W": power["gpu0_W"] if power else "",
            "gpu1_W": power["gpu1_W"] if power else "",
            "wall_W": wall_w,
            "system_W": power["system_W"] if power else "",
            "cpu_util_pct": power["cpu_util_pct"] if power else "",
            "efficiency_per_watt": efficiency,
            "power_samples": power["n_samples"] if power else 0,
        }
        results.append(result)

    # 출력 CSV
    output_path = args.output or str(data_dir / "throughput_summary.csv")
    fieldnames = [
        "exp_id", "workload", "cores", "rep",
        "throughput", "throughput_unit",
        "rapl_pkg_W", "rapl_dram_W", "gpu0_W", "gpu1_W", "wall_W", "system_W",
        "cpu_util_pct", "efficiency_per_watt", "power_samples",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            # None → 빈 문자열
            row = {}
            for k, v in result.items():
                if v is None:
                    row[k] = ""
                elif isinstance(v, float):
                    row[k] = f"{v:.4f}"
                else:
                    row[k] = v
            writer.writerow(row)

    print(f"\n출력: {output_path}")
    print(f"총 {len(results)}행")

    # 요약 통계 출력
    print("\n" + "=" * 70)
    print("요약 (워크로드별 평균)")
    print("=" * 70)
    print(f"{'워크로드':<12} {'코어':<6} {'throughput':<14} {'단위':<8} "
          f"{'RAPL_pkg':<10} {'wall_W':<10} {'system_W':<10} {'효율':<12}")
    print("-" * 80)

    # 워크로드 × 코어별 평균
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in results:
        key = (r["workload"], r["cores"])
        grouped[key].append(r)

    for key in sorted(grouped.keys()):
        workload, cores = key
        group = grouped[key]

        tp_vals = [r["throughput"] for r in group if r["throughput"] is not None]
        eff_vals = [r["efficiency_per_watt"] for r in group
                    if r["efficiency_per_watt"] is not None]
        pkg_vals = [r["rapl_pkg_W"] for r in group
                    if r["rapl_pkg_W"] != "" and r["rapl_pkg_W"] is not None]
        sys_vals = [r["system_W"] for r in group
                    if r["system_W"] != "" and r["system_W"] is not None]
        wall_vals = [r["wall_W"] for r in group
                     if r["wall_W"] is not None]

        avg_tp = sum(tp_vals) / len(tp_vals) if tp_vals else 0
        avg_eff = sum(eff_vals) / len(eff_vals) if eff_vals else 0
        avg_pkg = sum(float(v) for v in pkg_vals) / len(pkg_vals) if pkg_vals else 0
        avg_sys = sum(float(v) for v in sys_vals) / len(sys_vals) if sys_vals else 0
        avg_wall = sum(wall_vals) / len(wall_vals) if wall_vals else 0
        unit = group[0]["throughput_unit"]

        wall_str = f"{avg_wall:<10.1f}" if avg_wall > 0 else f"{'N/A':<10}"
        print(f"{workload:<12} {cores:<6} {avg_tp:<14.2f} {unit:<8} "
              f"{avg_pkg:<10.1f} {wall_str} {avg_sys:<10.1f} {avg_eff:<12.4f}")

    print("=" * 70)

    # 스케일링 분석 (3코어 대비 비율)
    print("\n" + "=" * 70)
    print("스케일링 분석 (3코어 대비 throughput 배율)")
    print("=" * 70)

    workload_names = sorted(set(r["workload"] for r in results))
    core_values = sorted(set(r["cores"] for r in results))

    print(f"{'워크로드':<12}", end="")
    for c in core_values:
        print(f"{'x' + str(c) + 'c':<8}", end="")
    print()
    print("-" * (12 + 8 * len(core_values)))

    for wl in workload_names:
        base_key = (wl, str(core_values[0]) if core_values else "3")
        base_group = grouped.get((wl, base_key), [])
        # Find base throughput (lowest core count)
        base_tp = None
        for c in core_values:
            key = (wl, str(c))
            group = grouped.get(key, [])
            if not group:
                key = (wl, c)
                group = grouped.get(key, [])
            if group:
                tp_vals = [r["throughput"] for r in group if r["throughput"] is not None]
                if tp_vals:
                    base_tp = sum(tp_vals) / len(tp_vals)
                    break

        if base_tp is None or base_tp == 0:
            continue

        print(f"{wl:<12}", end="")
        for c in core_values:
            key = (wl, str(c))
            group = grouped.get(key, [])
            if not group:
                key = (wl, c)
                group = grouped.get(key, [])
            if group:
                tp_vals = [r["throughput"] for r in group if r["throughput"] is not None]
                if tp_vals:
                    avg = sum(tp_vals) / len(tp_vals)
                    ratio = avg / base_tp
                    print(f"{ratio:<8.2f}", end="")
                else:
                    print(f"{'N/A':<8}", end="")
            else:
                print(f"{'N/A':<8}", end="")
        print()

    print("=" * 70)


if __name__ == "__main__":
    main()
