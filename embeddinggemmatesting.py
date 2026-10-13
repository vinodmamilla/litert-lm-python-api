#!/usr/bin/env python3
"""LiteRT-LM Multimodal Embedding Model (embedding-gemma-v2.litertlm) Testing Suite.

This script tests:
  1. Single Embedding Generation & Latency Benchmark for:
     - Text
     - Vision (Image)
     - Audio
     on CPU and Apple Silicon Metal GPU.

  2. Intra-Modal & Cross-Modal Verification using 3 Inputs (2 Same Kind, 1 Opposite):
     - Text:   Anchor Text vs Similar Text (Positive) vs Opposite Text (Negative)
     - Vision: Anchor Image vs Similar Image (Positive) vs Opposite Image (Negative)
     - Audio:  Anchor Audio vs Similar Audio (Positive) vs Opposite Audio (Negative)
     - Cross-Modal: Text Query vs Matching Media vs Opposite Media
"""

import argparse
import math
import os
import sys
import time
from typing import Any, Sequence

try:
    from litert_lm import Backend, EmbeddingEngine, Content
except ImportError:
    print(
        "ERROR: 'litert_lm' is not installed in the current Python environment.\n"
        "Please install it using:\n"
        "  pip install litert-lm\n"
        "or install the pre-release wheel:\n"
        "  pip install <path-to-litert-lm-wheel>.whl\n"
    )
    sys.exit(1)


def cosine_similarity(v1: Sequence[float], v2: Sequence[float]) -> float:
    """Computes cosine similarity between two float vectors."""
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def get_backend_instance(name: str, threads: int = 4, dispatch_lib_dir: str | None = None) -> Backend:
    """Returns the requested LiteRT Backend instance."""
    name = name.lower()
    if name == "gpu":
        return Backend.GPU()
    elif name == "cpu":
        return Backend.CPU(thread_count=threads)
    elif name == "npu":
        try:
            return Backend.NPU(litert_dispatch_lib_dir=dispatch_lib_dir)
        except RuntimeError as e:
            print(f"\n❌ NPU Backend Error: {e}")
            if sys.platform == "darwin":
                print("ℹ️ Note: On macOS, LiteRT-LM does not currently support Apple Neural Engine (ANE) via NPU.")
                print("   Please use '--backend gpu' (Apple Metal), which is fully supported and delivers 2.5x - 3x acceleration.")
            sys.exit(1)
    else:
        raise ValueError(f"Unsupported backend '{name}'. Choose 'cpu', 'gpu', 'npu', or 'both'.")


def find_embedding_model(explicit_path: str | None = None) -> str:
    """Finds the embedding-gemma-v2.litertlm model."""
    if explicit_path:
        expanded = os.path.abspath(os.path.expanduser(explicit_path))
        if os.path.exists(expanded):
            return expanded
        print(f"ERROR: Specified model file does not exist: {expanded}")
        sys.exit(1)

    env_path = os.environ.get("LITERTLM_EMBEDDING_MODEL")
    if env_path and os.path.exists(os.path.abspath(os.path.expanduser(env_path))):
        return os.path.abspath(os.path.expanduser(env_path))

    candidates = [
        os.path.join(os.getcwd(), "models", "embedding-gemma-v2.litertlm"),
        os.path.join(os.getcwd(), "embedding-gemma-v2.litertlm"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "embedding-gemma-v2.litertlm"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "embedding-gemma-v2.litertlm"),
    ]

    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)

    print("ERROR: Could not locate 'embedding-gemma-v2.litertlm' model file.")
    print("Please either:")
    print("  1. Pass the path: python3 embeddinggemmatesting.py --model /path/to/embedding-gemma-v2.litertlm")
    print("  2. Place it in ./models/ or the current directory")
    print("  3. Set environment variable: export LITERTLM_EMBEDDING_MODEL=/path/to/model")
    sys.exit(1)


def find_sample_directory(explicit_path: str | None = None) -> str:
    """Finds the samples/ directory across common candidate locations."""
    if explicit_path and os.path.exists(explicit_path):
        return os.path.abspath(explicit_path)

    candidates = [
        os.path.join(os.getcwd(), "samples"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples"),
    ]

    for c in candidates:
        if os.path.exists(os.path.join(c, "vision")) and os.path.exists(os.path.join(c, "audio")):
            return os.path.abspath(c)

    print("ERROR: Could not locate 'samples/' directory with vision/ and audio/ subfolders.")
    print("Checked locations:\n  " + "\n  ".join(candidates))
    sys.exit(1)


def resolve_sample_files(samples_dir: str) -> dict[str, str]:
    """Validates the 3 vision and 3 audio sample files."""
    v_dir = os.path.join(samples_dir, "vision")
    a_dir = os.path.join(samples_dir, "audio")

    files = {
        "apple_anchor": os.path.join(v_dir, "apple.png"),
        "apple_similar": os.path.join(v_dir, "apple_similar.png"),
        "car_opposite": os.path.join(v_dir, "blue_car.png"),
        "audio_anchor": os.path.join(a_dir, "have_a_wonderful_day.wav"),
        "audio_similar": os.path.join(a_dir, "have_a_wonderful_day_similar.wav"),
        "audio_opposite": os.path.join(a_dir, "synthetic_tone_440hz.wav"),
    }

    missing = [k for k, p in files.items() if not os.path.exists(p)]
    if missing:
        print(f"ERROR: Missing sample files in {samples_dir}:")
        for k in missing:
            print(f"  - {k}: {files[k]}")
        sys.exit(1)

    return files


# ==============================================================================
# 1. Single Embedding Generation & Benchmark
# ==============================================================================

def benchmark_single(
    engine: EmbeddingEngine,
    label: str,
    item_input: Any,
    iterations: int = 3,
) -> dict[str, Any]:
    """Generates a single embedding, inspects dimension, and measures latency."""
    # Warmup / Initial generation
    resp = engine.compute_embedding(item_input)
    vec = resp.embedding
    dim = len(vec)
    preview = vec[:4]

    # Latency benchmark
    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        engine.compute_embedding(item_input)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    avg_ms = sum(latencies) / len(latencies)
    return {
        "modality": label,
        "dim": dim,
        "preview": preview,
        "avg_ms": avg_ms,
        "embedding": vec,
    }


def run_single_embeddings_section(
    engine: EmbeddingEngine,
    backend_label: str,
    samples: dict[str, str],
    iterations: int = 3,
) -> dict[str, Any]:
    print("\n" + "=" * 80)
    print(f"  PART 1: SINGLE EMBEDDING GENERATION & BENCHMARK [{backend_label}]")
    print("=" * 80)

    # 1. Text single embedding
    sample_text = "LiteRT high-performance multimodal embeddings on Apple Silicon."
    print(f"\n[1] Single Text Embedding:")
    print(f"    Input: \"{sample_text}\"")
    r_text = benchmark_single(engine, "Text", sample_text, iterations=iterations)
    print(f"    ✓ Output Dimension: {r_text['dim']}")
    print(f"    ✓ Vector Preview:   {[round(x, 4) for x in r_text['preview']]} ...")
    print(f"    ✓ Avg Latency:      {r_text['avg_ms']:.2f} ms")

    # 2. Vision single embedding
    sample_img = samples["apple_anchor"]
    print(f"\n[2] Single Vision (Image) Embedding:")
    print(f"    Input: {os.path.basename(sample_img)} ({os.path.getsize(sample_img)} bytes)")
    r_img = benchmark_single(engine, "Vision", Content.ImageFile(sample_img), iterations=iterations)
    print(f"    ✓ Output Dimension: {r_img['dim']}")
    print(f"    ✓ Vector Preview:   {[round(x, 4) for x in r_img['preview']]} ...")
    print(f"    ✓ Avg Latency:      {r_img['avg_ms']:.2f} ms")

    # 3. Audio single embedding
    sample_aud = samples["audio_anchor"]
    print(f"\n[3] Single Audio Embedding:")
    print(f"    Input: {os.path.basename(sample_aud)} ({os.path.getsize(sample_aud)} bytes)")
    r_aud = benchmark_single(engine, "Audio", Content.AudioFile(sample_aud), iterations=iterations)
    print(f"    ✓ Output Dimension: {r_aud['dim']}")
    print(f"    ✓ Vector Preview:   {[round(x, 4) for x in r_aud['preview']]} ...")
    print(f"    ✓ Avg Latency:      {r_aud['avg_ms']:.2f} ms")

    return {"text": r_text, "vision": r_img, "audio": r_aud}


# ==============================================================================
# 2. Verification using 3 Inputs (2 Same Kind, 1 Opposite)
# ==============================================================================

def run_3_input_verification_section(
    engine: EmbeddingEngine,
    backend_label: str,
    samples: dict[str, str],
) -> bool:
    print("\n" + "=" * 80)
    print(f"  PART 2: 3-INPUT SIMILARITY VERIFICATION (2 Same Kind vs 1 Opposite) [{backend_label}]")
    print("=" * 80)

    all_correct = True

    # --------------------------------------------------------------------------
    # Test A: Text (3 Inputs: Anchor, Same Kind / Similar, Opposite / Different)
    # --------------------------------------------------------------------------
    print("\n[Test A] TEXT: 2 Same Kind vs 1 Opposite")
    t1_anchor = "Apples are delicious and nutritious round fruits."
    t2_similar = "A crisp, fresh red apple is great for a healthy snack."
    t3_opposite = "Quantum computers process information using quantum bits and superposition."

    emb_t1 = engine.compute_embedding(t1_anchor).embedding
    emb_t2 = engine.compute_embedding(t2_similar).embedding
    emb_t3 = engine.compute_embedding(t3_opposite).embedding

    sim_t_same = cosine_similarity(emb_t1, emb_t2)
    sim_t_opp = cosine_similarity(emb_t1, emb_t3)
    is_text_correct = sim_t_same > sim_t_opp
    all_correct = all_correct and is_text_correct

    print(f"  Input 1 (Anchor):        \"{t1_anchor}\"")
    print(f"  Input 2 (Same Kind):     \"{t2_similar}\"")
    print(f"  Input 3 (Opposite/Diff): \"{t3_opposite}\"")
    print(f"  --> Similarity(Input 1, Input 2 [Same]):     {sim_t_same:.4f}  (Expected HIGH)")
    print(f"  --> Similarity(Input 1, Input 3 [Opposite]): {sim_t_opp:.4f}  (Expected LOW)")
    verdict_text = "✅ CORRECT (PASS)" if is_text_correct else "❌ INCORRECT (FAIL)"
    print(f"  VERIFICATION [Sim(Same) > Sim(Opposite)]: {verdict_text} (Δ = +{sim_t_same - sim_t_opp:.4f})")

    # --------------------------------------------------------------------------
    # Test B: Vision (3 Inputs: Anchor Image, Same Kind Image, Opposite Image)
    # --------------------------------------------------------------------------
    print("\n[Test B] VISION: 2 Same Kind vs 1 Opposite")
    v1_anchor = samples["apple_anchor"]
    v2_similar = samples["apple_similar"]
    v3_opposite = samples["car_opposite"]

    emb_v1 = engine.compute_embedding(Content.ImageFile(v1_anchor)).embedding
    emb_v2 = engine.compute_embedding(Content.ImageFile(v2_similar)).embedding
    emb_v3 = engine.compute_embedding(Content.ImageFile(v3_opposite)).embedding

    sim_v_same = cosine_similarity(emb_v1, emb_v2)
    sim_v_opp = cosine_similarity(emb_v1, emb_v3)
    is_vision_correct = sim_v_same > sim_v_opp
    all_correct = all_correct and is_vision_correct

    print(f"  Input 1 (Anchor Image):        {os.path.basename(v1_anchor)} (Apple)")
    print(f"  Input 2 (Same Kind Image):     {os.path.basename(v2_similar)} (Apple Variant)")
    print(f"  Input 3 (Opposite/Diff Image): {os.path.basename(v3_opposite)} (Blue Car)")
    print(f"  --> Similarity(Image 1, Image 2 [Same]):     {sim_v_same:.4f}  (Expected HIGH)")
    print(f"  --> Similarity(Image 1, Image 3 [Opposite]): {sim_v_opp:.4f}  (Expected LOW)")
    verdict_vis = "✅ CORRECT (PASS)" if is_vision_correct else "❌ INCORRECT (FAIL)"
    print(f"  VERIFICATION [Sim(Same) > Sim(Opposite)]: {verdict_vis} (Δ = +{sim_v_same - sim_v_opp:.4f})")

    # --------------------------------------------------------------------------
    # Test C: Audio (3 Inputs: Anchor Audio, Same Kind Audio, Opposite Audio)
    # --------------------------------------------------------------------------
    print("\n[Test C] AUDIO: 2 Same Kind vs 1 Opposite")
    a1_anchor = samples["audio_anchor"]
    a2_similar = samples["audio_similar"]
    a3_opposite = samples["audio_opposite"]

    emb_a1 = engine.compute_embedding(Content.AudioFile(a1_anchor)).embedding
    emb_a2 = engine.compute_embedding(Content.AudioFile(a2_similar)).embedding
    emb_a3 = engine.compute_embedding(Content.AudioFile(a3_opposite)).embedding

    sim_a_same = cosine_similarity(emb_a1, emb_a2)
    sim_a_opp = cosine_similarity(emb_a1, emb_a3)
    is_audio_correct = sim_a_same > sim_a_opp
    all_correct = all_correct and is_audio_correct

    print(f"  Input 1 (Anchor Audio):        {os.path.basename(a1_anchor)} (Speech: 'Have a wonderful day')")
    print(f"  Input 2 (Same Kind Audio):     {os.path.basename(a2_similar)} (Speech: 'Have a wonderful day' variant)")
    print(f"  Input 3 (Opposite/Diff Audio): {os.path.basename(a3_opposite)} (Synthetic: 440Hz Pure Sine Tone)")
    print(f"  --> Similarity(Audio 1, Audio 2 [Same]):     {sim_a_same:.4f}  (Expected HIGH)")
    print(f"  --> Similarity(Audio 1, Audio 3 [Opposite]): {sim_a_opp:.4f}  (Expected LOW)")
    verdict_aud = "✅ CORRECT (PASS)" if is_audio_correct else "❌ INCORRECT (FAIL)"
    print(f"  VERIFICATION [Sim(Same) > Sim(Opposite)]: {verdict_aud} (Δ = +{sim_a_same - sim_a_opp:.4f})")

    # --------------------------------------------------------------------------
    # Test D: Cross-Modal Verification (Text Query vs Matching Media vs Opposite)
    # --------------------------------------------------------------------------
    print("\n[Test D] CROSS-MODAL: 1 Text Query vs 1 Matching Media vs 1 Opposite Media")

    # Text vs Images
    q_apple = "A fresh red apple fruit"
    emb_q_apple = engine.compute_embedding(q_apple).embedding
    sim_tv_match = cosine_similarity(emb_q_apple, emb_v1)
    sim_tv_opp = cosine_similarity(emb_q_apple, emb_v3)
    is_tv_correct = sim_tv_match > sim_tv_opp
    all_correct = all_correct and is_tv_correct

    print(f"  (1) Text Query: \"{q_apple}\"")
    print(f"      vs Matching Image (apple.png):   Score: {sim_tv_match:.4f}")
    print(f"      vs Opposite Image (blue_car.png): Score: {sim_tv_opp:.4f}")
    verdict_tv = "✅ CORRECT (PASS)" if is_tv_correct else "❌ INCORRECT (FAIL)"
    print(f"      VERIFICATION: {verdict_tv} (Δ = +{sim_tv_match - sim_tv_opp:.4f})")

    # Text vs Audios
    q_day = "Wishing you to have a wonderful day"
    emb_q_day = engine.compute_embedding(q_day).embedding
    sim_ta_match = cosine_similarity(emb_q_day, emb_a1)
    sim_ta_opp = cosine_similarity(emb_q_day, emb_a3)
    is_ta_correct = sim_ta_match > sim_ta_opp
    all_correct = all_correct and is_ta_correct

    print(f"  (2) Text Query: \"{q_day}\"")
    print(f"      vs Matching Audio (have_a_wonderful_day.wav): Score: {sim_ta_match:.4f}")
    print(f"      vs Opposite Audio (synthetic_tone_440hz.wav):  Score: {sim_ta_opp:.4f}")
    verdict_ta = "✅ CORRECT (PASS)" if is_ta_correct else "❌ INCORRECT (FAIL)"
    print(f"      VERIFICATION: {verdict_ta} (Δ = +{sim_ta_match - sim_ta_opp:.4f})")

    print("\n" + "-" * 80)
    overall_status = "✅ ALL 4 VERIFICATION TESTS PASSED (Text, Vision, Audio, Cross-Modal)!" if all_correct else "❌ SOME TESTS FAILED!"
    print(f"  Overall Result [{backend_label}]: {overall_status}")
    print("-" * 80)

    return all_correct


def main():
    parser = argparse.ArgumentParser(
        description="LiteRT-LM Gemma Embedding Benchmark & 3-Input Verification Suite"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to .litertlm model file (auto-discovered if omitted)",
    )
    parser.add_argument(
        "--samples-dir",
        type=str,
        default=None,
        help="Path to samples/ directory containing vision/ and audio/ subfolders",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["cpu", "gpu", "npu", "both"],
        default="both",
        help="Execution backend: 'cpu', 'gpu', or 'both' (default: both)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="CPU thread count (default: 4)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Benchmark iterations for single embeddings (default: 3)",
    )

    args = parser.parse_args()

    model_path = find_embedding_model(args.model)

    samples_dir = find_sample_directory(args.samples_dir)
    samples = resolve_sample_files(samples_dir)

    print("=" * 80)
    print("  LiteRT-LM Gemma Multimodal Embedding Evaluation")
    print(f"  Model:       {model_path}")
    print(f"  Samples Dir: {samples_dir}")
    print(f"  Backend:     {args.backend.upper()}")
    print("=" * 80)

    backends_to_run = ["cpu", "gpu"] if args.backend == "both" else [args.backend]
    bench_records: dict[str, dict[str, Any]] = {}

    for b_name in backends_to_run:
        label = b_name.upper()
        b_obj = get_backend_instance(b_name, threads=args.threads)

        with EmbeddingEngine(
            model_path=model_path,
            backend=b_obj,
            vision_backend=b_obj,
            audio_backend=b_obj,
        ) as engine:
            single_results = run_single_embeddings_section(
                engine, label, samples, iterations=args.iterations
            )
            bench_records[b_name] = single_results

            run_3_input_verification_section(engine, label, samples)

    # Print summary speedup table if both backends were tested
    if args.backend == "both" and "cpu" in bench_records and "gpu" in bench_records:
        print("\n" + "=" * 80)
        print("  HARDWARE COMPARISON: CPU vs GPU (Apple Metal Acceleration)")
        print("=" * 80)
        print(f"{'Modality':<10} | {'CPU Latency':<14} | {'GPU Latency':<14} | {'Speedup':<12}")
        print("-" * 80)
        for key in ["text", "vision", "audio"]:
            cpu_ms = bench_records["cpu"][key]["avg_ms"]
            gpu_ms = bench_records["gpu"][key]["avg_ms"]
            speedup = (cpu_ms / gpu_ms) if gpu_ms > 0 else 1.0
            print(
                f"{key.capitalize():<10} | {cpu_ms:>10.2f} ms   | {gpu_ms:>10.2f} ms   | {speedup:>8.2f}x faster"
            )
        print("-" * 80)


if __name__ == "__main__":
    main()
