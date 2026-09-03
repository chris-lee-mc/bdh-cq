# Reproduction run: identical logic to bdh-cq/figure7.py, just with finer-grained
# loss logging (every 50 steps instead of 200) and a smaller step budget for a
# CPU-only ~10-15 minute reproduction. All model/task/training/eval code is
# imported unmodified from the bdh_cq package.
import random
import time
from itertools import pairwise

import torch
from bdh_cq.bdh_cq import BDHReasoningWrapper
from bdh_cq.icq import (
    CLASS_WEIGHTS,
    cell_stats,
    decode_grid,
    generate_answer,
    ingest,
    make_model,
    task_at_level,
    task_prompt,
    train_loss,
)
from bdh_cq.tasks import TASKS

REASONING_STEPS_SWEEP = [1, 2, 4, 6, 8]
MAX_REASONING_STEPS = 8

SIZES = {"propagation": 5, "copy": 2, "order": 4, "nesting": 3}

LEVELS = {"propagation": [2, 3, 4], "copy": [2, 3], "order": [3, 4], "nesting": [2, 3]}


def run(
    device="cpu",
    family="order",
    steps=300,
    seed=3,
    write_prompt_to_memory=True,
    latent_step_embed=False,
    log_every=50,
):
    torch.manual_seed(seed)
    random.seed(seed)

    if device == "cpu":
        torch.set_num_threads(4)

    wrapper = BDHReasoningWrapper(
        make_model(
            dim=256,
            depth=4,
            dim_qk_heads=1024,
            attn_residual=True,
            attn_residual_depth_bias_distance=1,
        ),
        latent_step_embed=latent_step_embed,
    ).to(device)

    num_params = sum(p.numel() for p in wrapper.parameters())
    print(f"param count: {num_params}", flush=True)

    opt = torch.optim.AdamW(wrapper.parameters(), lr=1e-3, weight_decay=0.1)
    rng = random.Random(seed)

    t0 = time.time()
    loss_curve = []

    for step in range(steps):
        task_family = rng.choice(list(TASKS.values()))
        task = task_family(size=SIZES[task_family.name]).generate(seed=rng.randrange(2**31))

        reasoning_steps = rng.randint(0, MAX_REASONING_STEPS)

        loss = train_loss(
            wrapper,
            task,
            reasoning_steps,
            class_weights=CLASS_WEIGHTS,
            update_memory=write_prompt_to_memory,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(wrapper.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

        if step % log_every == 0 or step == steps - 1:
            elapsed = time.time() - t0
            print(f"step {step:5d}  loss {loss.item():.4f}  elapsed {elapsed:7.1f}s", flush=True)
            loss_curve.append((step, loss.item(), elapsed))

    train_time = time.time() - t0
    print(f"\ntotal training wall-clock: {train_time:.1f}s for {steps} steps", flush=True)

    wrapper.eval()

    task_family = TASKS[family]

    eval_t0 = time.time()
    with torch.no_grad():
        exact_match = {s: 0 for s in REASONING_STEPS_SWEEP}
        correct_cells = {s: 0 for s in REASONING_STEPS_SWEEP}
        total_cells = {s: 0 for s in REASONING_STEPS_SWEEP}
        num_outputs = 0

        for level in LEVELS[family]:
            for task_index in range(4):
                task = task_at_level(
                    task_family,
                    seed=1_000_000 + task_index * 10_000 + level,
                    level=level,
                    n_tests=2,
                    size=SIZES[family],
                )

                memories = ingest(wrapper, task_prompt(task), update_memory=write_prompt_to_memory)

                for _, _, target in task["test"]:
                    num_outputs += 1

                    for reasoning_steps in REASONING_STEPS_SWEEP:
                        predicted = generate_answer(
                            wrapper, task, reasoning_steps, memories=memories
                        )
                        predicted = decode_grid(predicted)

                        exact_match[reasoning_steps] += predicted.shape == target.shape and bool(
                            (predicted == target).all()
                        )
                        correct, total, _ = cell_stats(predicted, target)
                        correct_cells[reasoning_steps] += correct
                        total_cells[reasoning_steps] += total

    eval_time = time.time() - eval_t0
    print(f"eval wall-clock: {eval_time:.1f}s", flush=True)

    print()
    print(f"== {family}, {num_outputs} held-out outputs at each reasoning step")
    print(f"{'steps':<7}" + "".join(f"{s:>8}" for s in REASONING_STEPS_SWEEP))
    print(
        f"{'exact':<7}"
        + "".join(f"{exact_match[s]}/{num_outputs:<6}" for s in REASONING_STEPS_SWEEP)
    )
    print(
        f"{'cells':<7}"
        + "".join(
            f"{correct_cells[s] / max(1, total_cells[s]) * 100:7.1f}%"
            for s in REASONING_STEPS_SWEEP
        )
    )

    cell_accuracy = [correct_cells[s] / max(1, total_cells[s]) for s in REASONING_STEPS_SWEEP]
    print("monotone in R:", all(b >= a for a, b in pairwise(cell_accuracy)))
    print(f"\ntotal wall-clock (train+eval): {time.time() - t0:.1f}s")


if __name__ == "__main__":
    run()
