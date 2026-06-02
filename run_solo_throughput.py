#!/usr/bin/env python3
"""run_solo_throughput.py - UROP Solo Throughput 실험 자동화

9종 AI 워크로드에 CPU 코어를 3/5/7/10/14/17개로 바꿔가며
처리량(throughput) + 전력(power)을 동시 측정하여 에너지 효율 스케일링 패턴 도출.

기존 run_solo_experiments.py 대비 핵심 변경:
  1. OMP_NUM_THREADS 주입 (PyTorch 스레드 제어)
  2. stdout 캡처 (throughput 수집)
  3. 6개 코어 조건 × 3회 반복

실행 위치: gpu 서버 (203.255.176.80)
사전 준비:
  1. cgroup 그룹 생성 완료 (/sys/fs/cgroup/optimus/vm_a)
  2. sudo 권한 필요

사용법:
  sudo python3 -u run_solo_throughput.py 2>&1 | tee logs/throughput_run.log
  sudo python3 -u run_solo_throughput.py --workloads resnet --cores 3 --reps 1
  sudo python3 -u run_solo_throughput.py --dry-run
  sudo python3 -u run_solo_throughput.py --resume-from 42
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
CGROUP_VM = os.path.join(CGROUP_BASE, "vm_a")

# ==========================================
# 코어 조건 (6가지)
# ==========================================
CORE_CONDITIONS = {
    3:  {"cpuset": "0-2",       "omp": 3},
    5:  {"cpuset": "0-4",       "omp": 5},
    6:  {"cpuset": "0-5",       "omp": 6},
    7:  {"cpuset": "0-6",       "omp": 7},
    10: {"cpuset": "0-9",       "omp": 10},
    14: {"cpuset": "0-9,10-13", "omp": 14},   # 10 physical + 4 HT
    17: {"cpuset": "0-9,10-16", "omp": 17},   # 10 physical + 7 HT
}

DEFAULT_MEM = "28G"

# ==========================================
# 워크로드 정의 (14종)
# ==========================================
WORKLOADS = {
    "resnet": {
        "cmd": f"{VENV_PYTHON} {SCRIPTS_DIR}/workload_resnet.py -t {WORKLOAD_DUR}",
        "pattern": "workload_resnet.py",  # .py 포함 (resnet_gpu 오매칭 방지)
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
        "pattern": "workload_training.py",  # .py 포함 (training_heavy 오매칭 방지)
        "gpu": True,
        "throughput_unit": "img/s",
    },
    "llm": {
        "cmd": f"{VENV_PYTHON} {SCRIPTS_DIR}/workload_llm.py -t {WORKLOAD_DUR}",
        "pattern": "workload_llm.py",  # .py 포함 (gpu_llm 오매칭 방지)
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
        "direct_ffmpeg": True,  # Python wrapper 우회, 직접 ffmpeg 호출
    },
    "nodejs": {
        "cmd_server": f"node {HOME}/node/server.js",
        "cmd_load": "autocannon -c 100 -d {dur} http://localhost:3000",
        "pattern": "server.js",
        "two_phase": True,
        "gpu": False,
        "throughput_unit": "req/s",
    },
    # === 신규 GPU pipeline 워크로드 (Claim 2: GPU 세분화) ===
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

# 실험 순서 (GPU 워크로드 먼저 → CPU)
WORKLOAD_ORDER = [
    "resnet_gpu", "gemm", "yolo", "training", "gpu_llm",
    "training_heavy", "video_analytics", "llm_serving",
    "resnet", "llm", "llm_774m", "ffmpeg", "nodejs",
]

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
def setup_cgroup(cpus, mem):
    try:
        with open(os.path.join(CGROUP_VM, "cpuset.cpus"), "w") as f:
            f.write(cpus)
        with open(os.path.join(CGROUP_VM, "memory.max"), "w") as f:
            f.write(mem)
        log(f"  cgroup: cpus={cpus}, mem={mem}")
        return True
    except Exception as e:
        log(f"ERROR: cgroup 설정 실패: {e}")
        return False


def reset_cgroup():
    try:
        with open(os.path.join(CGROUP_VM, "cpuset.cpus"), "w") as f:
            f.write("0-19")
        with open(os.path.join(CGROUP_VM, "memory.max"), "w") as f:
            f.write("max")
    except Exception:
        pass


# ==========================================
# 프로세스 관리
# ==========================================
def stop_processes(procs):
    # SIGINT 먼저 → KeyboardInterrupt 핸들러가 "Done." 출력 + flush
    for proc in procs:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        except (ProcessLookupError, OSError):
            pass
    # "Done." 출력할 시간 대기
    time.sleep(3)
    # 아직 살아있으면 SIGTERM → SIGKILL
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
    patterns = ["server.js", "workload_yolo", "workload_gemm",
                "workload_resnet", "workload_llm", "autocannon",
                "workload_training", "libx264", "workload_gpu_llm",
                "ffmpeg", "workload_ffmpeg"]
    my_pid = os.getpid()
    parent_pid = os.getppid()
    killed = []
    for entry in os.listdir('/proc'):
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid in (my_pid, parent_pid):
            continue
        try:
            with open(f'/proc/{pid}/cmdline', 'rb') as f:
                cmdline = f.read().replace(b'\x00', b' ').decode('utf-8', errors='replace')
            if 'concurrent_power' in cmdline or 'run_solo_throughput' in cmdline:
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
# 워크로드 시작 (OMP_NUM_THREADS + stdout 캡처)
# ==========================================
def start_workload(workload_name, num_cores, stdout_path, stderr_path):
    """워크로드를 시작하고 cgroup에 등록. stdout을 파일로 캡처."""
    wl = WORKLOADS[workload_name]
    stdout_f = open(stdout_path, "w")
    stderr_f = open(stderr_path, "w")

    # 환경변수 설정: OMP_NUM_THREADS 주입 + stdout 버퍼링 방지
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["OMP_NUM_THREADS"] = str(num_cores)
    env["MKL_NUM_THREADS"] = str(num_cores)
    env["MKL_DYNAMIC"] = "FALSE"
    env["OPENBLAS_NUM_THREADS"] = str(num_cores)
    env["PYTHONUNBUFFERED"] = "1"

    if wl.get("two_phase"):
        # NodeJS: 서버 + autocannon
        server_proc = subprocess.Popen(
            ["bash", "-c", wl["cmd_server"]],
            stdout=subprocess.DEVNULL, stderr=stderr_f,
            preexec_fn=os.setsid, env=env,
        )
        with open(os.path.join(CGROUP_VM, "cgroup.procs"), "w") as f:
            f.write(str(server_proc.pid))
        log(f"  NodeJS 서버 시작 (PID={server_proc.pid})")
        time.sleep(2)

        load_cmd = wl["cmd_load"].format(dur=WORKLOAD_DUR)
        load_proc = subprocess.Popen(
            ["bash", "-c", load_cmd],
            stdout=stdout_f, stderr=stderr_f,
            preexec_fn=os.setsid, env=env,
        )
        with open(os.path.join(CGROUP_VM, "cgroup.procs"), "w") as f:
            f.write(str(load_proc.pid))
        log(f"  autocannon 시작 (PID={load_proc.pid})")
        return [server_proc, load_proc], stdout_f, stderr_f

    else:
        cmd = wl["cmd"]
        # GPU 워크로드: --device cuda:0
        if wl.get("gpu"):
            cmd += " --device cuda:0"
            log(f"  --device cuda:0")

        proc = subprocess.Popen(
            ["bash", "-c", cmd],
            stdout=stdout_f, stderr=stderr_f,
            preexec_fn=os.setsid, env=env,
        )
        with open(os.path.join(CGROUP_VM, "cgroup.procs"), "w") as f:
            f.write(str(proc.pid))
        log(f"  {workload_name} 시작 (PID={proc.pid}, OMP={num_cores})")
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

    # ffmpeg: -progress pipe:1 output에서 fps 파싱
    if wl.get("direct_ffmpeg"):
        if content.strip():
            return _parse_ffmpeg_progress(content), unit
        return None, unit

    # nodejs: autocannon stdout 먼저, 실패하면 stderr fallback
    if wl.get("two_phase"):
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

    # 일반 워크로드: "Done." 라인에서 throughput 파싱
    return _parse_done_line(content, workload_name), unit


def _parse_done_line(content, workload_name):
    """일반 워크로드의 "Done." 라인에서 throughput 추출.

    패턴 예시:
      resnet: "Done. 1234 images in 120.0s (10.3 img/s)"
      gemm:   "Done. 100 iterations in 120.0s (0.8 iter/s)"
      yolo:   "Done. 3000 frames in 120.0s (25.0 fps)"
      llm:    "Done. 50 iterations, 12345 tokens in 120.0s"
      gpu_llm: "Done. 50 iterations, 12345 tokens in 120.0s (102.9 tok/s)"
      training: "Done. 5000 images, 100 steps, 2 epochs in 120.0s (41.7 img/s)"
    """
    # 괄호 안의 값 파싱 시도 (img/s, iter/s, fps, tok/s)
    # 패턴: (숫자 단위) 형태
    match = re.search(r"Done\..*\((\d+\.?\d*)\s+\w+/s\)", content)
    if match:
        return float(match.group(1))

    # fps 패턴: (25.0 fps)
    match = re.search(r"Done\..*\((\d+\.?\d*)\s+fps\)", content)
    if match:
        return float(match.group(1))

    # llm 패턴 (괄호 없음): "Done. N iterations, M tokens in S.Ss"
    # throughput = tokens / seconds
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
    """ffmpeg -progress pipe:1 출력에서 평균 fps 추출.

    -progress 출력 형식:
      frame=123
      fps=25.5
      ...
      progress=continue
    마지막 fps= 라인의 값을 사용.
    """
    fps_values = re.findall(r"^fps=(\d+\.?\d*)$", content, re.MULTILINE)
    if not fps_values:
        return None

    # 마지막 보고된 fps (안정 상태)
    valid_fps = [float(v) for v in fps_values if float(v) > 0]
    if not valid_fps:
        return None

    # 처음 20%는 워밍업 제외, 나머지의 평균
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
# 단일 실험 실행
# ==========================================
def run_single_experiment(workload_name, num_cores, rep, mem, log_dir, dry_run=False):
    """단일 실험 실행: 1 워크로드 × 1 코어 조건 × 1 반복.

    Returns:
        dict with experiment results, or None on failure.
    """
    cond = CORE_CONDITIONS[num_cores]
    cpuset = cond["cpuset"]
    omp = cond["omp"]
    exp_tag = f"{workload_name}_{num_cores}c_r{rep}"

    log_sep()
    log(f"실험: {exp_tag} (cpuset={cpuset}, OMP={omp}, mem={mem})")
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
    if not setup_cgroup(cpuset, mem):
        return None

    # 파일 경로 설정
    power_dir = os.path.join(log_dir, "power")
    stdout_dir = os.path.join(log_dir, "stdout")
    stderr_dir = os.path.join(log_dir, "stderr")
    os.makedirs(power_dir, exist_ok=True)
    os.makedirs(stdout_dir, exist_ok=True)
    os.makedirs(stderr_dir, exist_ok=True)

    power_csv = os.path.join(power_dir, f"{exp_tag}.csv")
    stdout_path = os.path.join(stdout_dir, f"{exp_tag}.log")
    stderr_path = os.path.join(stderr_dir, f"{exp_tag}.err")

    # 3. concurrent_power.py 시작
    pattern = WORKLOADS[workload_name]["pattern"]
    conc_cmd = [
        "python3", CONC_POWER_PY,
        "--workloads", f"vm_a:{pattern}",
        "-i", "1", "-t", str(TOTAL_MEASURE),
        "-o", power_csv,
    ]

    ts_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log(f">>> 시작: {ts_start}")
    log(f"  power CSV: {power_csv}")
    log(f"  stdout: {stdout_path}")

    if dry_run:
        log(f"[DRY-RUN] {exp_tag}")
        reset_cgroup()
        return {
            "exp_tag": exp_tag,
            "workload": workload_name,
            "cores": num_cores,
            "cpuset": cpuset,
            "omp_threads": omp,
            "mem": mem,
            "rep": rep,
            "start_time": ts_start,
            "end_time": ts_start,
            "throughput": None,
            "throughput_unit": WORKLOADS[workload_name]["throughput_unit"],
        }

    conc_proc = subprocess.Popen(
        conc_cmd,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    # 4. idle 구간
    log(f"idle 구간 ({IDLE_WAIT}초)...")
    time.sleep(IDLE_WAIT)

    # 5. 워크로드 시작
    log("워크로드 시작...")
    procs, stdout_f, stderr_f = start_workload(
        workload_name, omp, stdout_path, stderr_path
    )

    log(f"워크로드 실행 ({WORKLOAD_DUR}초)...")
    time.sleep(WORKLOAD_DUR)

    # 6. 워크로드 종료
    log("워크로드 종료...")
    stop_processes(procs)
    stdout_f.close()
    stderr_f.close()

    # 7. 쿨다운
    log(f"쿨다운 ({COOLDOWN}초)...")
    time.sleep(COOLDOWN)

    # 8. 측정 종료
    try:
        conc_proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        conc_proc.terminate()
        conc_proc.wait(timeout=5)

    ts_end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log(f">>> 종료: {ts_end}")

    # 9. cgroup 초기화
    reset_cgroup()

    # 10. throughput 파싱 (nodejs는 stderr fallback)
    throughput, unit = parse_throughput_from_log(
        stdout_path, workload_name, stderr_path=stderr_path)
    if throughput is not None:
        log(f"  throughput: {throughput:.2f} {unit}")
    else:
        log(f"  WARNING: throughput 파싱 실패 (stdout: {stdout_path})")

    return {
        "exp_tag": exp_tag,
        "workload": workload_name,
        "cores": num_cores,
        "cpuset": cpuset,
        "omp_threads": omp,
        "mem": mem,
        "rep": rep,
        "start_time": ts_start,
        "end_time": ts_end,
        "throughput": throughput,
        "throughput_unit": unit,
    }


# ==========================================
# 메인
# ==========================================
def main():
    parser = argparse.ArgumentParser(
        description="UROP Solo Throughput 실험 (11 워크로드 × 6 코어 × 3 reps)")
    parser.add_argument("--workloads", nargs="+", default=None,
                        help="워크로드 선택 (기본: 전체)")
    parser.add_argument("--cores", nargs="+", type=int, default=None,
                        help="코어 조건 선택 (기본: 3 5 7 10 14 17)")
    parser.add_argument("--reps", type=int, default=3,
                        help="반복 횟수 (기본: 3)")
    parser.add_argument("--mem", default=DEFAULT_MEM,
                        help=f"메모리 제한 (기본: {DEFAULT_MEM})")
    parser.add_argument("--dry-run", action="store_true",
                        help="테스트 모드")
    parser.add_argument("--resume-from", type=int, default=None,
                        help="중단 지점부터 재개 (exp_id 번호)")
    args = parser.parse_args()

    # 워크로드 목록
    if args.workloads:
        workloads = args.workloads
        for wl in workloads:
            if wl not in WORKLOADS:
                print(f"ERROR: 알 수 없는 워크로드: {wl}")
                print(f"  가능한 값: {', '.join(WORKLOADS.keys())}")
                sys.exit(1)
    else:
        workloads = WORKLOAD_ORDER[:]

    # 코어 조건
    if args.cores:
        cores_list = args.cores
        for c in cores_list:
            if c not in CORE_CONDITIONS:
                print(f"ERROR: 알 수 없는 코어 조건: {c}")
                print(f"  가능한 값: {', '.join(map(str, CORE_CONDITIONS.keys()))}")
                sys.exit(1)
    else:
        cores_list = sorted(CORE_CONDITIONS.keys())

    # sudo 확인
    if os.geteuid() != 0 and not args.dry_run:
        print("ERROR: sudo 권한이 필요합니다.")
        print("  sudo python3 -u run_solo_throughput.py")
        sys.exit(1)

    # cgroup 확인
    if not os.path.isdir(CGROUP_VM) and not args.dry_run:
        print(f"ERROR: cgroup 디렉토리 없음: {CGROUP_VM}")
        print(f"  sudo mkdir -p {CGROUP_VM}")
        sys.exit(1)

    # 출력 디렉토리 생성
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M")
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "logs", timestamp_str)
    os.makedirs(log_dir, exist_ok=True)

    # 로그 파일
    global _log_file
    log_path = os.path.join(log_dir, "experiment.log")
    _log_file = open(log_path, "a")

    # 실험 매트릭스 생성
    experiments = []
    exp_id = 0
    for wl_name in workloads:
        for num_cores in cores_list:
            for rep in range(1, args.reps + 1):
                exp_id += 1
                experiments.append({
                    "id": exp_id,
                    "workload": wl_name,
                    "cores": num_cores,
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
    log(f"UROP Solo Throughput 실험")
    log(f"워크로드: {', '.join(workloads)}")
    log(f"코어 조건: {cores_list}")
    log(f"반복: {args.reps}회")
    log(f"메모리: {args.mem}")
    log(f"프로토콜: {IDLE_WAIT}s idle + {WORKLOAD_DUR}s 워크로드 + {COOLDOWN}s 쿨다운")
    log(f"총 실험: {total_exps}개 (전체 {len(experiments)}개 중)")
    if args.dry_run:
        log("[DRY-RUN 모드]")
    est_min = total_exps * (TOTAL_MEASURE + 60) / 60
    log(f"예상 소요: ~{est_min:.0f}분 (~{est_min/60:.1f}시간)")
    log(f"출력: {log_dir}")
    log_sep()

    # CSV 로그 파일 생성
    ts_csv_path = os.path.join(log_dir, "solo_timestamps.csv")
    csv_exists = os.path.exists(ts_csv_path)
    ts_csv = open(ts_csv_path, "a", newline="")
    ts_writer = csv.writer(ts_csv)
    if not csv_exists:
        ts_writer.writerow([
            "exp_id", "workload", "cores", "cpuset", "omp_threads", "mem",
            "rep", "start_time", "end_time", "throughput", "throughput_unit",
        ])

    # 실험 실행
    success_count = 0
    for i in range(start_idx, len(experiments)):
        exp = experiments[i]
        log(f"\n[{i - start_idx + 1}/{total_exps}] "
            f"exp_id={exp['id']}: {exp['workload']} "
            f"{exp['cores']}c rep{exp['rep']}")

        result = run_single_experiment(
            workload_name=exp["workload"],
            num_cores=exp["cores"],
            rep=exp["rep"],
            mem=args.mem,
            log_dir=log_dir,
            dry_run=args.dry_run,
        )

        if result is not None:
            success_count += 1
            ts_writer.writerow([
                exp["id"],
                result["workload"],
                result["cores"],
                result["cpuset"],
                result["omp_threads"],
                result["mem"],
                result["rep"],
                result["start_time"],
                result["end_time"],
                result["throughput"] if result["throughput"] is not None else "",
                result["throughput_unit"],
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
    print(f"  2. python3 parse_throughput.py --data-dir <local_path>")


if __name__ == "__main__":
    main()
