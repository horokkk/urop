#!/usr/bin/env python3
"""workload_training_heavy.py - GPU Training with Heavy Data Pipeline
실행: python3 workload_training_heavy.py -t 120 --device cuda:0
목적: Heavy augmentation + 많은 DataLoader workers로 CPU 전처리 부하를 높여
      CPU 코어 수가 GPU training throughput에 미치는 영향을 측정.

기존 workload_training.py 대비 변경:
- transforms: RandomResizedCrop, ColorJitter, RandomRotation, GaussianBlur, RandomErasing 추가
- DataLoader num_workers: 2 → 8 (기본값)
- 나머지 (모델, 배치, 학습 루프) 동일
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import time
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description='GPU Training with heavy data pipeline')
    parser.add_argument('-t', '--timeout', type=int, default=120,
                        help='Duration in seconds (default: 120)')
    parser.add_argument('-b', '--batch', type=int, default=64,
                        help='Batch size (default: 64)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='CUDA device (default: cuda:0)')
    parser.add_argument('--workers', type=int, default=8,
                        help='DataLoader num_workers (default: 8)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save training logs (default: None)')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return

    device = torch.device(args.device)
    dev_idx = device.index if device.index is not None else 0
    print(f"Device: {torch.cuda.get_device_name(dev_idx)}")
    print(f"Batch size: {args.batch}")
    print(f"DataLoader workers: {args.workers}")
    print(f"Duration: {args.timeout}s")
    print(f"Pipeline: HEAVY (RandomResizedCrop + ColorJitter + Rotation + GaussianBlur + RandomErasing)")

    # Heavy augmentation pipeline (CPU-intensive)
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33)),
    ])

    print("Loading CIFAR-10 dataset...")
    data_dir = os.path.join(os.path.expanduser("~"), ".cache", "cifar10")
    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
        persistent_workers=True)

    # ResNet-18 pretrained fine-tune (기존과 동일)
    print("Loading ResNet-18 (pretrained, fine-tune mode)...")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    # Warmup
    warmup_input = torch.randn(args.batch, 3, 224, 224, device=device)
    warmup_target = torch.zeros(args.batch, dtype=torch.long, device=device)
    output = model(warmup_input)
    loss = criterion(output, warmup_target)
    loss.backward()
    optimizer.zero_grad()
    torch.cuda.synchronize()

    mem_alloc = torch.cuda.memory_allocated(dev_idx) / (1024**3)
    print(f"GPU memory after warmup: {mem_alloc:.2f} GB")
    print("Starting training (heavy pipeline)...\n")

    start_time = time.time()
    epoch = 0
    total_steps = 0
    total_images = 0

    try:
        while time.time() - start_time < args.timeout:
            epoch += 1
            for inputs, targets in trainloader:
                if time.time() - start_time >= args.timeout:
                    break

                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                total_steps += 1
                total_images += inputs.size(0)

                if total_steps % 20 == 0:
                    elapsed = time.time() - start_time
                    ips = total_images / elapsed
                    print(f"  [{elapsed:.1f}s] epoch {epoch} step {total_steps} | "
                          f"loss={loss.item():.4f} | {ips:.1f} img/s")

    except KeyboardInterrupt:
        pass

    torch.cuda.synchronize()

    elapsed = time.time() - start_time
    mem_peak = torch.cuda.max_memory_allocated(dev_idx) / (1024**3)
    print(f"\nDone. {total_images} images, {total_steps} steps, {epoch} epochs "
          f"in {elapsed:.1f}s ({total_images/elapsed:.1f} img/s)")
    print(f"GPU peak memory: {mem_peak:.2f} GB")


if __name__ == '__main__':
    main()
