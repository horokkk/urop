#!/usr/bin/env python3
"""workload_llm.py - CPU+Memory 지배적 워크로드 (LLM Tokenization + Small Inference)
실행: python3 workload_llm.py -t 120
목적: CPU와 메모리에 부하를 걸어서 DRAM 전력 비율이 높은 워크로드 생성

변경 (P5 해결): 구간별 throughput 측정 추가
- 5초 구간마다 토큰 처리량 기록 → 자연스러운 분산 확보
- Done 줄에 구간 평균 throughput 출력 (기존 total/elapsed 대신)

사전 설치: pip install transformers
"""

import torch
import time
import argparse
import os
import numpy as np


def check_transformers():
    try:
        import transformers
        return True
    except ImportError:
        print("ERROR: transformers not installed")
        print("Run: pip install transformers")
        return False


def main():
    parser = argparse.ArgumentParser(description='CPU+Memory dominant LLM workload')
    parser.add_argument('-t', '--timeout', type=int, default=120,
                        help='Duration in seconds (default: 120)')
    parser.add_argument('--model', type=str, default='gpt2',
                        help='Model name (default: gpt2)')
    parser.add_argument('--interval', type=float, default=5.0,
                        help='Throughput measurement interval in seconds (default: 5.0)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save generated texts (default: None)')
    args = parser.parse_args()

    if not check_transformers():
        return

    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = torch.device('cpu')
    print(f"Device: CPU only")
    print(f"Model: {args.model}")
    print(f"Duration: {args.timeout}s")
    print(f"Measurement interval: {args.interval}s")

    # 모델 & 토크나이저 로드
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()
    model.to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 긴 텍스트 생성 (토크나이저에 부하)
    base_text = (
        "Artificial intelligence is transforming every aspect of modern society. "
        "From healthcare to transportation, machine learning algorithms are being deployed "
        "to solve complex problems that were previously considered intractable. "
        "Deep neural networks have shown remarkable capabilities in understanding natural language, "
        "recognizing images, and generating creative content. "
    )
    long_text = base_text * 50  # 긴 텍스트로 토크나이저 부하

    # 결과 저장 준비
    save_results = args.output_dir is not None
    result_file = None
    if save_results:
        os.makedirs(args.output_dir, exist_ok=True)
        result_path = os.path.join(args.output_dir, "llm_results.txt")
        result_file = open(result_path, "w")
        print(f"Saving results to: {result_path}")

    print("Starting LLM workload (tokenization + inference)...\n")

    start_time = time.time()
    iteration = 0
    total_tokens_processed = 0

    # 구간별 throughput 측정
    interval_throughputs = []
    interval_start = start_time
    interval_tokens = 0
    warmup_done = False

    try:
        with torch.no_grad():
            while time.time() - start_time < args.timeout:
                # Phase 1: 대량 토크나이제이션 (CPU + Memory 집약)
                tokens = tokenizer(long_text, return_tensors='pt',
                                   truncation=True, max_length=512,
                                   padding='max_length')
                tokens_this_iter = tokens['input_ids'].shape[1]
                total_tokens_processed += tokens_this_iter
                interval_tokens += tokens_this_iter

                # Phase 2: 모델 추론 (CPU + Memory 집약)
                input_ids = tokens['input_ids'].to(device)
                attention_mask = tokens['attention_mask'].to(device)
                output = model(input_ids=input_ids, attention_mask=attention_mask)

                # Phase 3: 텍스트 생성 (순차적 디코딩 - 메모리 집약)
                generated = model.generate(
                    input_ids[:, :20],  # 짧은 프롬프트에서 생성
                    max_new_tokens=50,
                    do_sample=False
                )
                gen_tokens = generated.shape[1]
                total_tokens_processed += gen_tokens
                interval_tokens += gen_tokens

                # 결과 저장: 생성된 텍스트
                if save_results:
                    text = tokenizer.decode(generated[0], skip_special_tokens=True)
                    result_file.write(f"--- iteration {iteration + 1} ---\n")
                    result_file.write(text + "\n\n")

                iteration += 1

                # 구간 체크
                now = time.time()
                interval_elapsed = now - interval_start
                if interval_elapsed >= args.interval:
                    interval_tps = interval_tokens / interval_elapsed
                    # 첫 구간은 warmup으로 제외
                    if warmup_done:
                        interval_throughputs.append(interval_tps)
                    else:
                        warmup_done = True
                    total_elapsed = now - start_time
                    print(f"  [{total_elapsed:.1f}s] iter {iteration} | "
                          f"interval: {interval_tps:.1f} tok/s | "
                          f"tokens: {total_tokens_processed}")
                    interval_start = now
                    interval_tokens = 0

    except KeyboardInterrupt:
        pass

    # 마지막 미완료 구간 처리
    now = time.time()
    remaining_elapsed = now - interval_start
    if remaining_elapsed > 1.0 and interval_tokens > 0 and warmup_done:
        interval_throughputs.append(interval_tokens / remaining_elapsed)

    if result_file:
        result_file.close()

    elapsed = time.time() - start_time

    # 구간 평균/표준편차 계산
    if interval_throughputs:
        avg_tps = np.mean(interval_throughputs)
        std_tps = np.std(interval_throughputs)
        print(f"\nDone. {iteration} iterations, {total_tokens_processed} tokens "
              f"in {elapsed:.3f}s ({avg_tps:.1f} tok/s) [std={std_tps:.2f}, n={len(interval_throughputs)}]")
    else:
        avg_tps = total_tokens_processed / elapsed if elapsed > 0 else 0
        print(f"\nDone. {iteration} iterations, {total_tokens_processed} tokens "
              f"in {elapsed:.3f}s ({avg_tps:.1f} tok/s)")


if __name__ == '__main__':
    main()
