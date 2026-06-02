#!/usr/bin/env python3
"""run_concurrent_throughput.py - UROP Concurrent Throughput 실험 자동화

Claim 3 (유형 인지 배치 → 에너지 절감) 검증을 위한 concurrent 실험.
핵심 질문: Solo throughput 프로파일이 concurrent 실행에서도 유지되는가?

vm_a + vm_b 동시 실행, 양쪽 throughput + power 동시 측정.
run_solo_throughput.py 기반, concurrent 전용으로 수정.

실행 위치: gpu 서버 (203.255.176.80)
사전 준비:
  1. cgroup 그룹 생성 완료 (/sys/fs/cgroup/optimus/vm_a, vm_b)
  2. sudo 권한 필요

사용법:
  sudo python3 -u run_concurrent_throughput.py \
      --pairs "ffmpeg:training" "resnet:llm" "ffmpeg:nodejs" \
      --splits "10:10" "14:6" "17:3" \
      --reps 3

  sudo python3 -u run_concurrent_throughput.py --dry-run
  sudo python3 -u run_concurrent_throughput.py --resume-from 5
"""

import subprocess
import time
import os
import sys
import signal
import argparse
import csv
import re
from datetime import datetime
from pathlib import Path

# ==========================================
# 설정
# ==========================================
HOME = "/home/optimus"
SCRIPTS_DIR = f"{HOME}/jiyoon_energy/scripts"
CONC_POWER_PY = os.path.join(SCRIPTS_DIR, "concurrent_power.py")
VENV_PYTHON = f"{HOME}/yolo/venv/bin/python3"

# 실험 프로토콜
IDLE_WAIT = 30
WORKLOAD_DUR = 120
COOLDOWN = 30
TOTAL_MEASURE = IDLE_WAIT + WORKLOAD_DUR + COOLDOWN  # 180s

# idle 감지 설정
IDLE_CPU_TH = 5.0
IDLE_GPU_TH = 5.0
IDLE_STABLE_SEC = 5
IDLE_TIMEOUT = 120

# cgroup 경로
CGROUP_BASE = "/sys/fs/cgroup/optimus"
CGROUP_A = os.path.join(CGROUP_BASE, "vm_a")
CGROUP_B = os.path.join(CGROUP_BASE, "vm_b")

# 총 스레드 수 (10 physical + 10 HT)
TOTAL_THREADS = 20
# 총 메모리 (GB)
TOTAL_MEM_GB = 28

# ==========================================
# 워크로드 정의 (run_solo_throughput.py와 동일)
# ==========================================
WORKLOADS = {
    "resnet": {
        "cmd": f"{VENV_PYTHON} {SCRIPTS_DIR}/workload_resnet.py -t {WORKLOAD_DUR}",
        "pattern": "workload_resnet.py",
        "gpu": False,
        "throughput_unit": "img/s",
    },
    "resnet_gpu": {
        "cmd": f"{VENV_PYTHON} {SCRIPTS_DIR}/workload_resnet_gpu.py -t {WORKLOAD_DUR}",
        "pattern": "workload_resnet_gpu",
        "gpu": True,
        "throughput_unit": "img/s",
    },
    "training": {
        "cmd": f"{VENV_PYTHON} {SCRIPTS_DIR}/workload_training.py -t {WORKLOAD_DUR}",
        "pattern": "workload_training.py",
        "gpu": True,
        "throughput_unit": "img/s",
    },
    "llm": {
        "cmd": f"{VENV_PYTHON} {SCRIPTS_DIR}/workload_llm.py -t {WORKLOAD_DUR}",
        "pattern": "workload_llm.py",
        "gpu": False,
        "throughput_unit": "tok/s",
    },
    "llm_774m": {
        "cmd": f"{VENV_PYTHON} {SCRIPTS_DIR}/workload_llm.py -t {WORKLOAD_DUR} --model gpt2-medium",
        "pattern": "workload_llm.py",
        "gpu": False,
        "throughput_unit": "tok/s",
    },
    "gpu_llm": {
        "cmd": f"{VENV_PYTHON} {SCRIPTS_DIR}/workload_gpu_llm.py -t {WORKLOAD_DUR}",
        "pattern": "workload_gpu_llm",
        "gpu": True,
        "throughput_unit": "tok/s",
    },
    "yolo": {
        "cmd": f"{VENV_PYTHON} {SCRIPTS_DIR}/workload_yolo.py -t {WORKLOAD_DUR} --source {HOME}/yolo/test_video.mp4",
        "pattern": "workload_yolo",
        "gpu": True,
        "throughput_unit": "fps",
    },
    "gemm": {
        "cmd": f"{VENV_PYTHON} {SCRIPTS_DIR}/workload_gemm.py -t {WORKLOAD_DUR}",
        "pattern": "workload_gemm",
        "gpu": True,
        "throughput_unit": "iter/s",
    },
    "ffmpeg": {
        "cmd": (
            "ffmpeg -nostdin -loglevel error "
            "-f lavfi -i testsrc2=duration=9999:size=1920x1080:rate=30 "
            "-c:v libx264 -preset medium "
            "-progress pipe:1 "
            "-f null -y /dev/null"
        ),
        "pattern": "libx264",
        "gpu": False,
        "throughput_unit": "fps",
        "direct_ffmpeg": True,
    },
    "nodejs": {
        "cmd_server": f"node {HOME}/node/server.js",
        "cmd_load": "autocannon -c 100 -d {dur} http://localhost:3000",
        "pattern": "server.js",
        "two_phase": True,
        "gpu": False,
        "throughput_unit": "req/s",
    },
    "training_heavy": {
        "cmd": f"{VENV_PYTHON} {SCRIPTS_DIR}/workload_training_heavy.py -t {WORKLOAD_DUR}",
        "pattern": "workload_training_heavy",
        "gpu": True,
        "throughput_unit": "img/s",
    },
    "video_analytics": {
        "cmd": f"{VENV_PYTHON} {SCRIPTS_DIR}/workload_video_analytics.py -t {WORKLOAD_DUR} --source {HOME}/yolo/test_video.mp4",
        "pattern": "workload_video_analytics",
        "gpu": True,
        "throughput_unit": "fps",
    },
    "llm_serving": {
        "cmd": f"{VENV_PYTHON} {SCRIPTS_DIR}/workload_llm_serving.py -t {WORKLOAD_DUR}",
        "pattern": "workload_llm_serving",
        "gpu": True,
        "throughput_unit": "tok/s",
    },
}

# ==========================================
# 로깅
# ==========================================
_log_file = None


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    if _log_file:
        _log_file.write(line + "\n")
        _log_file.flush()


def log_sep():
    log("=" * 60)


# ==========================================
# 코어 분배
# ==========================================
def make_split(a_cores, b_cores):
    """코어 수를 cpuset 문자열 + 메모리 제한으로 변환.

    Args:
        a_cores: vm_a에 할당할 코어 수
        b_cores: vm_b에 할당할 코어 수

    Returns:
        (a_cpuset, b_cpuset, a_mem, b_mem)
    """
    assert a_cores + b_cores == TOTAL_THREADS, \
        f"a_cores({a_cores}) + b_cores({b_cores}) != {TOTAL_THREADS}"

    a_cpuset = f"0-{a_cores - 1}"
    b_cpuset = f"{a_cores}-{a_cores + b_cores - 1}"

    # 메모리: 비례 분할, 최소 4G
    a_mem_gb = max(4, int(TOTAL_MEM_GB * a_cores / TOTAL_THREADS))
    b_mem_gb = max(4, int(TOTAL_MEM_GB * b_cores / TOTAL_THREADS))
    a_mem = f"{a_mem_gb}G"
    b_mem = f"{b_mem_gb}G"

    return a_cpuset, b_cpuset, a_mem, b_mem


def parse_split(split_str):
    """'10:10' 형태 문자열을 (a_cores, b_cores) 튜플로 파싱."""
    parts = split_str.split(":")
    if len(parts) != 2:
        raise ValueError(f"잘못된 split 형식: {split_str} (예: '10:10')")
    a_cores, b_cores = int(parts[0]), int(parts[1])
    if a_cores + b_cores != TOTAL_THREADS:
        raise ValueError(
            f"split {split_str}: 합이 {a_cores + b_cores}이지만 "
            f"{TOTAL_THREADS}이어야 합니다")
    if a_cores < 1 or b_cores < 1:
        raise ValueError(f"split {split_str}: 각 VM에 최소 1코어 필요")
    return a_cores, b_cores


def parse_pair(pair_str):
    """'ffmpeg:training' 형태 문자열을 (wl_a, wl_b) 튜플로 파싱."""
    parts = pair_str.split(":")
    if len(parts) != 2:
        raise ValueError(f"잘못된 pair 형식: {pair_str} (예: 'ffmpeg:training')")
    wl_a, wl_b = parts[0], parts[1]
    if wl_a not in WORKLOADS:
        raise ValueError(f"알 수 없는 워크로드: {wl_a}")
    if wl_b not in WORKLOADS:
        raise ValueError(f"알 수 없는 워크로드: {wl_b}")
    return wl_a, wl_b


# ==========================================
# idle 감지
# ==========================================
def get_cpu_util():
    try:
        with open("/proc/stat") as f:
            line = f.readline()
        parts = line.split()
        times = [int(x) for x in parts[1:]]
        idle = times[3] + times[4]
        total = sum(times)
        return idle, total
    except Exception:
        return None, None


def get_gpu_util():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"],
            text=True, timeout=2
        ).strip()
        return float(out.split("\n")[0].strip())
    except Exception:
        return 0.0


def wait_for_idle():
    log(f"idle 감지 (CPU<{IDLE_CPU_TH}%, GPU<{IDLE_GPU_TH}%, "
        f"{IDLE_STABLE_SEC}초 안정)")

    stable = 0
    prev_idle, prev_total = get_cpu_util()
    time.sleep(1)
    start = time.time()

    while stable < IDLE_STABLE_SEC:
        if time.time() - start > IDLE_TIMEOUT:
            log(f"WARNING: idle 대기 {IDLE_TIMEOUT}초 초과 - 강제 진행")
            return False

        curr_idle, curr_total = get_cpu_util()
        if curr_idle is None or prev_total is None:
            time.sleep(1)
            continue

        d_idle = curr_idle - prev_idle
        d_total = curr_total - prev_total
        cpu_util = (1.0 - d_idle / d_total) * 100.0 if d_total > 0 else 0.0
        gpu_util = get_gpu_util()

        prev_idle, prev_total = curr_idle, curr_total

        if cpu_util < IDLE_CPU_TH and gpu_util < IDLE_GPU_TH:
            stable += 1
            log(f"  CPU={cpu_util:.1f}%, GPU={gpu_util:.0f}% "
                f"(stable {stable}/{IDLE_STABLE_SEC})")
        else:
            if stable > 0:
                log(f"  리셋: CPU={cpu_util:.1f}%, GPU={gpu_util:.0f}%")
            stable = 0

        time.sleep(1)

    log("idle 확인 완료")
    return True


# ==========================================
# cgroup 설정
# ==========================================
def setup_cgroup(cgroup_path, cpus, mem):
    """cgroup에 CPU 코어와 메모리 할당."""
    try:
        with open(os.path.join(cgroup_path, "cpuset.cpus"), "w") as f:
            f.write(cpus)
        with open(os.path.join(cgroup_path, "memory.max"), "w") as f:
            f.write(mem)
        vm_name = os.path.basename(cgroup_path)
        log(f"  {vm_name}: cpus={cpus}, mem={mem}")
        return True
    except Exception as e:
        log(f"ERROR: cgroup 설정 실패 ({cgroup_path}): {e}")
        return False


def reset_cgroup(cgroup_path):
    """cgroup을 기본값으로 초기화."""
    try:
        with open(os.path.join(cgroup_path, "cpuset.cpus"), "w") as f:
            f.write("0-19")
        with open(os.path.join(cgroup_path, "memory.max"), "w") as f:
            f.write("max")
    except Exception:
        pass


# ==========================================
# 프로세스 관리
# ==========================================
def stop_processes(procs):
    """프로세스 그룹 종료: SIGINT → SIGTERM → SIGKILL."""
    for proc in procs:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        except (ProcessLookupError, OSError):
            pass
    # "Done." 출력할 시간 대기
    time.sleep(3)
    for proc in procs:
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, OSError):
                pass
    deadline = time.time() + 5
    for proc in procs:
        remaining = max(0, deadline - time.time())
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass


def cleanup_residual_processes():
    """이전 실험의 잔재 프로세스 정리."""
    patterns = ["server.js", "workload_yolo", "workload_gemm",
                "workload_resnet", "workload_llm", "autocannon",
                "workload_training", "libx264", "workload_gpu_llm",
                "ffmpeg", "workload_ffmpeg",
                "workload_video_analytics", "workload_llm_serving",
                "workload_training_heavy"]
    my_pid = os.getpid()
    my_ppid = os.getppid()  # sudo 부모 프로세스 보호
    killed = []
    for entry in os.listdir('/proc'):
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid in (my_pid, my_ppid):
            continue
        try:
            with open(f'/proc/{pid}/cmdline', 'rb') as f:
                cmdline = f.read().replace(b'\x00', b' ').decode(
                    'utf-8', errors='replace')
            if 'concurrent_power' in cmdline:
                continue
            if 'run_concurrent_throughput' in cmdline:
                continue
            for pat in patterns:
                if pat in cmdline:
                    os.kill(pid, signal.SIGKILL)
                    killed.append((pid, pat))
                    break
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            continue
    if killed:
        log(f"  잔재 프로세스 제거: {killed}")
    return killed


# ==========================================
# 워크로드 시작 (cgroup_path + cuda_device 매개변수 추가)
# ==========================================
def start_workload(workload_name, num_cores, cgroup_path, cuda_device,
                   stdout_path, stderr_path):
    """워크로드를 시작하고 cgroup에 등록.

    Args:
        workload_name: 워크로드 이름
        num_cores: OMP_NUM_THREADS 값
        cgroup_path: cgroup 경로 (vm_a 또는 vm_b)
        cuda_device: GPU 디바이스 번호 (0 또는 1), GPU 워크로드에만 사용
        stdout_path: stdout 캡처 파일 경로
        stderr_path: stderr 캡처 파일 경로

    Returns:
        (procs, stdout_f, stderr_f)
    """
    wl = WORKLOADS[workload_name]
    stdout_f = open(stdout_path, "w")
    stderr_f = open(stderr_path, "w")

    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["OMP_NUM_THREADS"] = str(num_cores)
    env["MKL_NUM_THREADS"] = str(num_cores)
    env["MKL_DYNAMIC"] = "FALSE"
    env["OPENBLAS_NUM_THREADS"] = str(num_cores)
    env["PYTHONUNBUFFERED"] = "1"

    vm_name = os.path.basename(cgroup_path)

    if wl.get("two_phase"):
        # NodeJS: 서버 + autocannon
        server_proc = subprocess.Popen(
            ["bash", "-c", wl["cmd_server"]],
            stdout=subprocess.DEVNULL, stderr=stderr_f,
            preexec_fn=os.setsid, env=env,
        )
        with open(os.path.join(cgroup_path, "cgroup.procs"), "w") as f:
            f.write(str(server_proc.pid))
        log(f"  [{vm_name}] NodeJS 서버 시작 (PID={server_proc.pid})")
        time.sleep(2)

        load_cmd = wl["cmd_load"].format(dur=WORKLOAD_DUR)
        load_proc = subprocess.Popen(
            ["bash", "-c", load_cmd],
            stdout=stdout_f, stderr=stderr_f,
            preexec_fn=os.setsid, env=env,
        )
        with open(os.path.join(cgroup_path, "cgroup.procs"), "w") as f:
            f.write(str(load_proc.pid))
        log(f"  [{vm_name}] autocannon 시작 (PID={load_proc.pid})")
        return [server_proc, load_proc], stdout_f, stderr_f

    else:
        cmd = wl["cmd"]
        # GPU 워크로드: --device cuda:N
        if wl.get("gpu"):
            cmd += f" --device cuda:{cuda_device}"
            log(f"  [{vm_name}] --device cuda:{cuda_device}")

        proc = subprocess.Popen(
            ["bash", "-c", cmd],
            stdout=stdout_f, stderr=stderr_f,
            preexec_fn=os.setsid, env=env,
        )
        with open(os.path.join(cgroup_path, "cgroup.procs"), "w") as f:
            f.write(str(proc.pid))
        log(f"  [{vm_name}] {workload_name} 시작 "
            f"(PID={proc.pid}, OMP={num_cores})")
        return [proc], stdout_f, stderr_f


# ==========================================
# Throughput 파싱
# ==========================================
def parse_throughput_from_log(stdout_path, workload_name, stderr_path=None):
    """stdout (+ stderr fallback) 로그에서 throughput 값을 파싱.

    Args:
        stdout_path: stdout 로그 파일 경로
        workload_name: 워크로드 이름
        stderr_path: stderr 로그 파일 경로 (nodejs/autocannon fallback용)

    Returns:
        (throughput_value: float or None, unit: str)
    """
    wl = WORKLOADS[workload_name]
    unit = wl["throughput_unit"]

    try:
        with open(stdout_path, "r") as f:
            content = f.read()
    except FileNotFoundError:
        content = ""

    if wl.get("direct_ffmpeg"):
        if content.strip():
            return _parse_ffmpeg_progress(content), unit
        return None, unit

    if wl.get("two_phase"):
        # autocannon: stdout 먼저, 실패하면 stderr fallback
        result = _parse_autocannon(content) if content.strip() else None
        if result is None and stderr_path:
            try:
                with open(stderr_path, "r") as f:
                    stderr_content = f.read()
                if stderr_content.strip():
                    result = _parse_autocannon(stderr_content)
            except FileNotFoundError:
                pass
        return result, unit

    if not content.strip():
        return None, unit

    return _parse_done_line(content, workload_name), unit


def _parse_done_line(content, workload_name):
    """일반 워크로드의 'Done.' 라인에서 throughput 추출."""
    match = re.search(r"Done\..*\((\d+\.?\d*)\s+\w+/s\)", content)
    if match:
        return float(match.group(1))

    match = re.search(r"Done\..*\((\d+\.?\d*)\s+fps\)", content)
    if match:
        return float(match.group(1))

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


def _parse_ffmpeg_progress(content):
    """ffmpeg -progress pipe:1 출력에서 평균 fps 추출."""
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


def _strip_ansi(text):
    """ANSI escape 시퀀스 제거."""
    return re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', text)


def _parse_autocannon(content):
    """autocannon stdout/stderr에서 평균 Req/Sec 파싱."""
    content = _strip_ansi(content)

    # autocannon 테이블 모드: "Req/Sec" 행에서 Avg 열 (5번째 값) 추출
    # 형식: │ Req/Sec │ 1% │ 2.5% │ 50% │ 97.5% │ Avg │ Stdev │ Min │
    for line in content.split('\n'):
        if 'Req/Sec' in line and '\u2502' in line:
            parts = [p.strip() for p in line.split('\u2502') if p.strip()]
            if len(parts) >= 6:
                val_str = parts[5]
                try:
                    if val_str.endswith('k'):
                        return float(val_str[:-1]) * 1000
                    return float(val_str)
                except ValueError:
                    pass

    # "X req/sec" 형태 검색
    match = re.search(
        r"(\d+\.?\d*k?)\s+req(?:uest)?s?/sec", content, re.IGNORECASE)
    if match:
        val_str = match.group(1)
        if val_str.endswith('k'):
            return float(val_str[:-1]) * 1000
        return float(val_str)

    # 숫자만 들어있는 경우 (autocannon --json)
    match = re.search(r'"average"\s*:\s*(\d+\.?\d*)', content)
    if match:
        return float(match.group(1))

    return None


# ==========================================
# Concurrent 실험 실행
# ==========================================
def run_concurrent_experiment(wl_a, wl_b, a_cores, b_cores, rep,
                              log_dir, dry_run=False):
    """1회 concurrent 실험: wl_a(vm_a) + wl_b(vm_b) 동시 실행.

    Returns:
        dict with experiment results, or None on failure.
    """
    a_cpuset, b_cpuset, a_mem, b_mem = make_split(a_cores, b_cores)
    exp_tag = f"{wl_a}+{wl_b}_{a_cores}v{b_cores}_r{rep}"

    log_sep()
    log(f"실험: {exp_tag}")
    log(f"  vm_a: {wl_a} (cpuset={a_cpuset}, OMP={a_cores}, mem={a_mem})")
    log(f"  vm_b: {wl_b} (cpuset={b_cpuset}, OMP={b_cores}, mem={b_mem})")
    log_sep()

    # 0. 잔재 프로세스 정리
    cleanup_residual_processes()

    # 0.5. page cache 초기화
    try:
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3")
        log("  page cache 초기화")
    except Exception as e:
        log(f"  WARNING: drop_caches 실패: {e}")

    # 1. idle 감지
    wait_for_idle()

    # 2. cgroup 설정
    log("cgroup 설정...")
    if not setup_cgroup(CGROUP_A, a_cpuset, a_mem):
        return None
    if not setup_cgroup(CGROUP_B, b_cpuset, b_mem):
        return None

    # 파일 경로 설정
    power_dir = os.path.join(log_dir, "power")
    stdout_a_dir = os.path.join(log_dir, "stdout_a")
    stdout_b_dir = os.path.join(log_dir, "stdout_b")
    stderr_a_dir = os.path.join(log_dir, "stderr_a")
    stderr_b_dir = os.path.join(log_dir, "stderr_b")
    for d in [power_dir, stdout_a_dir, stdout_b_dir,
              stderr_a_dir, stderr_b_dir]:
        os.makedirs(d, exist_ok=True)

    power_csv = os.path.join(power_dir, f"{exp_tag}.csv")
    stdout_a_path = os.path.join(stdout_a_dir, f"{exp_tag}.log")
    stdout_b_path = os.path.join(stdout_b_dir, f"{exp_tag}.log")
    stderr_a_path = os.path.join(stderr_a_dir, f"{exp_tag}.err")
    stderr_b_path = os.path.join(stderr_b_dir, f"{exp_tag}.err")

    # 3. concurrent_power.py 시작 (양쪽 워크로드 모니터링)
    pattern_a = WORKLOADS[wl_a]["pattern"]
    pattern_b = WORKLOADS[wl_b]["pattern"]
    conc_cmd = [
        "python3", CONC_POWER_PY,
        "--workloads", f"vm_a:{pattern_a}", f"vm_b:{pattern_b}",
        "-i", "1", "-t", str(TOTAL_MEASURE),
        "-o", power_csv,
    ]

    ts_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log(f">>> 시작: {ts_start}")
    log(f"  power CSV: {power_csv}")
    log(f"  stdout_a: {stdout_a_path}")
    log(f"  stdout_b: {stdout_b_path}")

    if dry_run:
        log(f"[DRY-RUN] {exp_tag}")
        reset_cgroup(CGROUP_A)
        reset_cgroup(CGROUP_B)
        return {
            "exp_tag": exp_tag,
            "wl_a": wl_a, "wl_b": wl_b,
            "a_cores": a_cores, "b_cores": b_cores,
            "a_cpuset": a_cpuset, "b_cpuset": b_cpuset,
            "a_omp": a_cores, "b_omp": b_cores,
            "a_mem": a_mem, "b_mem": b_mem,
            "rep": rep,
            "start_time": ts_start, "end_time": ts_start,
            "throughput_a": None,
            "unit_a": WORKLOADS[wl_a]["throughput_unit"],
            "throughput_b": None,
            "unit_b": WORKLOADS[wl_b]["throughput_unit"],
        }

    conc_proc = subprocess.Popen(
        conc_cmd,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    # 4. idle 구간
    log(f"idle 구간 ({IDLE_WAIT}초)...")
    time.sleep(IDLE_WAIT)

    # 5. 워크로드 A 시작 (vm_a, cuda:0)
    log("워크로드 A 시작...")
    procs_a, stdout_a_f, stderr_a_f = start_workload(
        wl_a, a_cores, CGROUP_A, cuda_device=0,
        stdout_path=stdout_a_path, stderr_path=stderr_a_path,
    )

    # 6. 워크로드 B 시작 (vm_b, cuda:1)
    log("워크로드 B 시작...")
    procs_b, stdout_b_f, stderr_b_f = start_workload(
        wl_b, b_cores, CGROUP_B, cuda_device=1,
        stdout_path=stdout_b_path, stderr_path=stderr_b_path,
    )

    # 7. 워크로드 실행
    log(f"워크로드 실행 ({WORKLOAD_DUR}초)...")
    time.sleep(WORKLOAD_DUR)

    # 8. 워크로드 종료 (양쪽 동시)
    log("워크로드 종료...")
    stop_processes(procs_a + procs_b)
    stdout_a_f.close()
    stderr_a_f.close()
    stdout_b_f.close()
    stderr_b_f.close()

    # 9. 쿨다운
    log(f"쿨다운 ({COOLDOWN}초)...")
    time.sleep(COOLDOWN)

    # 10. 측정 종료
    try:
        conc_proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        conc_proc.terminate()
        conc_proc.wait(timeout=5)

    ts_end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log(f">>> 종료: {ts_end}")

    # 11. cgroup 초기화
    reset_cgroup(CGROUP_A)
    reset_cgroup(CGROUP_B)

    # 12. throughput 파싱 (nodejs는 stderr fallback)
    throughput_a, unit_a = parse_throughput_from_log(
        stdout_a_path, wl_a, stderr_path=stderr_a_path)
    throughput_b, unit_b = parse_throughput_from_log(
        stdout_b_path, wl_b, stderr_path=stderr_b_path)

    if throughput_a is not None:
        log(f"  throughput_a ({wl_a}): {throughput_a:.2f} {unit_a}")
    else:
        log(f"  WARNING: throughput_a 파싱 실패 (stdout: {stdout_a_path})")

    if throughput_b is not None:
        log(f"  throughput_b ({wl_b}): {throughput_b:.2f} {unit_b}")
    else:
        log(f"  WARNING: throughput_b 파싱 실패 (stdout: {stdout_b_path})")

    return {
        "exp_tag": exp_tag,
        "wl_a": wl_a, "wl_b": wl_b,
        "a_cores": a_cores, "b_cores": b_cores,
        "a_cpuset": a_cpuset, "b_cpuset": b_cpuset,
        "a_omp": a_cores, "b_omp": b_cores,
        "a_mem": a_mem, "b_mem": b_mem,
        "rep": rep,
        "start_time": ts_start, "end_time": ts_end,
        "throughput_a": throughput_a, "unit_a": unit_a,
        "throughput_b": throughput_b, "unit_b": unit_b,
    }


# ==========================================
# 메인
# ==========================================
def main():
    parser = argparse.ArgumentParser(
        description="UROP Concurrent Throughput 실험 "
                    "(워크로드 쌍 × 코어 분배 × 반복)")
    parser.add_argument(
        "--pairs", nargs="+", required=True,
        help="워크로드 쌍 (A:B), 예: 'ffmpeg:training' 'resnet:llm'")
    parser.add_argument(
        "--splits", nargs="+", default=["10:10", "14:6", "17:3", "6:14", "3:17"],
        help="코어 분배 (A:B, 합=20), 예: '10:10' '14:6' '17:3'")
    parser.add_argument(
        "--reps", type=int, default=3,
        help="반복 횟수 (기본: 3)")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="테스트 모드")
    parser.add_argument(
        "--resume-from", type=int, default=None,
        help="중단 지점부터 재개 (exp_id 번호)")
    args = parser.parse_args()

    # 입력 검증
    pairs = []
    for p in args.pairs:
        try:
            pairs.append(parse_pair(p))
        except ValueError as e:
            print(f"ERROR: {e}")
            print(f"  가능한 워크로드: {', '.join(sorted(WORKLOADS.keys()))}")
            sys.exit(1)

    splits = []
    for s in args.splits:
        try:
            splits.append(parse_split(s))
        except ValueError as e:
            print(f"ERROR: {e}")
            sys.exit(1)

    # GPU 충돌 검증: 양쪽 모두 GPU 사용 시 서로 다른 디바이스 사용 가능한지 확인
    for wl_a, wl_b in pairs:
        if WORKLOADS[wl_a].get("gpu") and WORKLOADS[wl_b].get("gpu"):
            log(f"NOTE: {wl_a}(cuda:0) + {wl_b}(cuda:1) — 양쪽 GPU 사용")

    # sudo 확인
    if os.geteuid() != 0 and not args.dry_run:
        print("ERROR: sudo 권한이 필요합니다.")
        print("  sudo python3 -u run_concurrent_throughput.py \\")
        print("      --pairs 'ffmpeg:training' --splits '10:10' --reps 1")
        sys.exit(1)

    # cgroup 확인
    for cg_path, cg_name in [(CGROUP_A, "vm_a"), (CGROUP_B, "vm_b")]:
        if not os.path.isdir(cg_path) and not args.dry_run:
            print(f"ERROR: cgroup 디렉토리 없음: {cg_path}")
            print(f"  sudo mkdir -p {cg_path}")
            sys.exit(1)

    # 출력 디렉토리 생성
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M")
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "logs", f"conc_{timestamp_str}")
    os.makedirs(log_dir, exist_ok=True)

    # 로그 파일
    global _log_file
    log_path = os.path.join(log_dir, "experiment.log")
    _log_file = open(log_path, "a")

    # 실험 매트릭스 생성
    experiments = []
    exp_id = 0
    for wl_a, wl_b in pairs:
        for a_cores, b_cores in splits:
            for rep in range(1, args.reps + 1):
                exp_id += 1
                experiments.append({
                    "id": exp_id,
                    "wl_a": wl_a,
                    "wl_b": wl_b,
                    "a_cores": a_cores,
                    "b_cores": b_cores,
                    "rep": rep,
                })

    # resume 처리
    start_idx = 0
    if args.resume_from is not None:
        for i, exp in enumerate(experiments):
            if exp["id"] >= args.resume_from:
                start_idx = i
                break
        log(f"resume: exp_id={args.resume_from}부터 재개 (idx={start_idx})")

    total_exps = len(experiments) - start_idx

    # 실험 목록 출력
    log_sep()
    log(f"UROP Concurrent Throughput 실험")
    log(f"워크로드 쌍: {[f'{a}:{b}' for a, b in pairs]}")
    log(f"코어 분배: {[f'{a}:{b}' for a, b in splits]}")
    log(f"반복: {args.reps}회")
    log(f"프로토콜: {IDLE_WAIT}s idle + {WORKLOAD_DUR}s 워크로드 + "
        f"{COOLDOWN}s 쿨다운")
    log(f"총 실험: {total_exps}개 (전체 {len(experiments)}개 중)")
    if args.dry_run:
        log("[DRY-RUN 모드]")
    est_min = total_exps * (TOTAL_MEASURE + 60) / 60
    log(f"예상 소요: ~{est_min:.0f}분 (~{est_min/60:.1f}시간)")
    log(f"출력: {log_dir}")
    log_sep()

    # CSV 로그 파일 생성
    ts_csv_path = os.path.join(log_dir, "concurrent_timestamps.csv")
    csv_exists = os.path.exists(ts_csv_path)
    ts_csv = open(ts_csv_path, "a", newline="")
    ts_writer = csv.writer(ts_csv)
    if not csv_exists:
        ts_writer.writerow([
            "exp_id", "wl_a", "wl_b",
            "a_cores", "b_cores", "a_cpuset", "b_cpuset",
            "a_omp", "b_omp", "a_mem", "b_mem",
            "rep", "start_time", "end_time",
            "throughput_a", "unit_a", "throughput_b", "unit_b",
        ])

    # 실험 실행
    success_count = 0
    for i in range(start_idx, len(experiments)):
        exp = experiments[i]
        log(f"\n[{i - start_idx + 1}/{total_exps}] "
            f"exp_id={exp['id']}: {exp['wl_a']}+{exp['wl_b']} "
            f"{exp['a_cores']}v{exp['b_cores']} rep{exp['rep']}")

        result = run_concurrent_experiment(
            wl_a=exp["wl_a"],
            wl_b=exp["wl_b"],
            a_cores=exp["a_cores"],
            b_cores=exp["b_cores"],
            rep=exp["rep"],
            log_dir=log_dir,
            dry_run=args.dry_run,
        )

        if result is not None:
            success_count += 1
            ts_writer.writerow([
                exp["id"],
                result["wl_a"], result["wl_b"],
                result["a_cores"], result["b_cores"],
                result["a_cpuset"], result["b_cpuset"],
                result["a_omp"], result["b_omp"],
                result["a_mem"], result["b_mem"],
                result["rep"],
                result["start_time"], result["end_time"],
                result["throughput_a"] if result["throughput_a"] is not None else "",
                result["unit_a"],
                result["throughput_b"] if result["throughput_b"] is not None else "",
                result["unit_b"],
            ])
            ts_csv.flush()
            log(f"  => 성공 (exp_id={exp['id']})")
        else:
            log(f"  => 실패 (exp_id={exp['id']})")

    # 완료
    log_sep()
    log(f"전체 완료: {success_count}/{total_exps} 성공")
    log(f"타임스탬프 CSV: {ts_csv_path}")
    log(f"로그: {log_path}")
    log_sep()

    ts_csv.close()
    _log_file.close()

    print(f"\n결과 디렉토리: {log_dir}")
    print(f"다음 단계:")
    print(f"  1. scp -P 4247 -r optimus@203.255.176.80:{log_dir} ~/urop/data/")
    print(f"  2. 결과 분석: concurrent throughput / solo throughput >= 0.9 확인")


if __name__ == "__main__":
    main()
