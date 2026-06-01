#!/usr/bin/env python3
"""workload_llm_serving.py - LLM Serving Simulation (CPU tokenization + GPU generation)
실행: python3 workload_llm_serving.py -t 120 --device cuda:0
목적: LLM serving 시나리오를 시뮬레이션하여 CPU 토크나이징/전처리가
      GPU generation throughput에 미치는 영향을 측정.

기존 workload_gpu_llm.py와의 차이:
- gpu_llm: 동일 프롬프트를 반복 → 토크나이징 부하 최소
- llm_serving: 다양한 프롬프트 배치를 매번 토크나이징 + 패딩 + 후처리
  → CPU 토크나이징/배치 구성 부하가 명시적으로 드러남

CPU 작업: 프롬프트 생성/샘플링 + batch tokenization + padding + output decoding
GPU 작업: autoregressive token generation
"""

import torch
import time
import argparse
import os
import random
import hashlib


# 다양한 길이/주제의 프롬프트 풀
PROMPT_POOL = [
    "Explain the concept of energy-efficient computing in modern data centers and how resource allocation strategies can minimize power consumption while maintaining performance.",
    "Write a detailed technical analysis of CPU-GPU interaction patterns in deep learning training pipelines, focusing on data loading bottlenecks.",
    "Describe the evolution of cloud computing from virtualization to containerization, including the implications for resource management and energy consumption.",
    "What are the key challenges in deploying large language models at scale? Discuss inference optimization techniques such as quantization, pruning, and knowledge distillation.",
    "Analyze the trade-offs between throughput and latency in real-time AI serving systems. How do batching strategies affect energy efficiency?",
    "Compare and contrast different CPU scheduling algorithms (CFS, FIFO, Round Robin) in the context of mixed AI workloads running on shared infrastructure.",
    "Explain how RAPL (Running Average Power Limit) works in Intel processors and discuss its accuracy for measuring CPU and DRAM power consumption.",
    "Discuss the impact of memory bandwidth on AI workload performance. How does the memory wall problem affect different types of neural network architectures?",
    "Write about the environmental impact of AI training and inference. What are the current efforts to make AI more sustainable?",
    "Describe how cgroup v2 resource isolation works in Linux and its implications for containerized AI workloads in Kubernetes clusters.",
    "Explain the architecture of modern GPU computing, focusing on CUDA cores, tensor cores, and HBM memory hierarchy.",
    "What is the relationship between instruction-level parallelism (IPC) and energy consumption in multi-core processors?",
    "Discuss the challenges of fair resource allocation in multi-tenant cloud environments running heterogeneous AI workloads.",
    "Analyze the energy consumption patterns of transformer-based models during different phases: prefill vs decode.",
    "How do dynamic voltage and frequency scaling (DVFS) techniques interact with AI workload characteristics?",
    "Explain the concept of compute-bound vs memory-bound workloads and how this classification affects optimal resource allocation.",
    "Describe the role of data augmentation in deep learning training and its impact on CPU utilization in the data loading pipeline.",
    "What are the key metrics for evaluating energy efficiency in AI inference serving? Discuss throughput per watt, latency, and total cost of ownership.",
    "Compare the energy characteristics of batch inference vs real-time serving for computer vision models.",
    "Discuss how hardware heterogeneity (different CPU, GPU, and accelerator types) affects energy-aware workload scheduling.",
]


def generate_dynamic_prompt(base_prompts, batch_size):
    """다양한 길이의 프롬프트 배치 생성 (CPU 작업)."""
    prompts = []
    for _ in range(batch_size):
        # 랜덤 프롬프트 선택
        p = random.choice(base_prompts)

        # 프롬프트 변형 (CPU 작업 추가)
        # 1. 랜덤 prefix 추가
        prefix = random.choice([
            "Please provide a comprehensive answer: ",
            "In 500 words or more, ",
            "As an expert in this field, ",
            "Considering recent developments, ",
            "From a systems perspective, ",
        ])

        # 2. 프롬프트 반복/확장 (길이 변형)
        repeat = random.randint(1, 3)
        p = prefix + (p + " ") * repeat

        # 3. 해시 기반 유니크 suffix (매번 다른 토큰 시퀀스 보장)
        suffix = hashlib.md5(f"{time.time()}_{random.random()}".encode()).hexdigest()[:16]
        p += f" [ref:{suffix}]"

        prompts.append(p)

    return prompts


def main():
    parser = argparse.ArgumentParser(description='LLM serving simulation workload')
    parser.add_argument('-t', '--timeout', type=int, default=120,
                        help='Duration in seconds (default: 120)')
    parser.add_argument('--model', type=str, default='facebook/opt-1.3b',
                        help='HuggingFace model (default: facebook/opt-1.3b)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='CUDA device (default: cuda:0)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for serving (default: 4)')
    parser.add_argument('--max-new-tokens', type=int, default=64,
                        help='Max tokens per generation (default: 64)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='(unused, for compatibility)')
    parser.add_argument('--cache-dir', type=str,
                        default='/data/home/optimus/huggingface_cache',
                        help='HuggingFace cache directory')
    args = parser.parse_args()

    os.environ['HF_HOME'] = args.cache_dir

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError:
        print("ERROR: transformers not installed")
        return

    device = torch.device(args.device)
    dev_idx = device.index if device.index is not None else 0
    print(f"Device: {torch.cuda.get_device_name(dev_idx)}")
    print(f"Model: {args.model}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Duration: {args.timeout}s")
    print(f"Pipeline: Batch tokenization + GPU generation + output decoding")

    # 모델 로드
    print("Loading model (float16)...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    model.to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # 배치 생성 시 left-padding 필요

    # Warmup
    print("Warming up...")
    warmup_input = tokenizer("Hello world", return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        model.generate(warmup_input, max_new_tokens=10, do_sample=False)
    torch.cuda.synchronize()

    mem_alloc = torch.cuda.memory_allocated(dev_idx) / (1024**3)
    print(f"GPU memory (model loaded): {mem_alloc:.2f} GB")
    print("Starting LLM serving simulation...\n")

    start_time = time.time()
    total_requests = 0
    total_tokens_generated = 0
    iteration = 0

    try:
        with torch.no_grad():
            while time.time() - start_time < args.timeout:
                iteration += 1

                # === CPU 작업: 프롬프트 생성 + 배치 토크나이징 ===
                prompts = generate_dynamic_prompt(PROMPT_POOL, args.batch_size)

                # 배치 토크나이징 (CPU-heavy: 다양한 길이 → padding 필요)
                inputs = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)

                # === GPU 작업: 배치 생성 ===
                outputs = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                )

                # === CPU 작업: 출력 디코딩 ===
                generated_tokens = outputs.shape[1] - input_ids.shape[1]
                total_tokens_generated += generated_tokens * args.batch_size
                total_requests += args.batch_size

                # 출력 텍스트 디코딩 (CPU 작업)
                for i in range(args.batch_size):
                    _ = tokenizer.decode(outputs[i], skip_special_tokens=True)

                if iteration % 3 == 0:
                    elapsed = time.time() - start_time
                    tps = total_tokens_generated / elapsed
                    rps = total_requests / elapsed
                    print(f"  [{elapsed:.1f}s] iter {iteration} | "
                          f"requests: {total_requests} ({rps:.1f} req/s) | "
                          f"tokens: {total_tokens_generated} ({tps:.1f} tok/s)")

    except KeyboardInterrupt:
        pass

    torch.cuda.synchronize()

    elapsed = time.time() - start_time
    mem_peak = torch.cuda.max_memory_allocated(dev_idx) / (1024**3)
    tps = total_tokens_generated / elapsed
    print(f"\nDone. {total_requests} requests, {total_tokens_generated} tokens "
          f"in {elapsed:.1f}s ({tps:.1f} tok/s)")
    print(f"GPU peak memory: {mem_peak:.2f} GB")


if __name__ == '__main__':
    main()
