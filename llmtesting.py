#!/usr/bin/env python3
"""LiteRT-LM Interactive Multi-Turn Chat & Testing CLI.

Usage:
  # Start interactive multi-turn chat on CPU:
  python3 llmtesting.py --model /path/to/gemma-4-E2B-it.litertlm --backend cpu

  # Start interactive multi-turn chat on GPU (Apple Metal / WebGPU):
  python3 llmtesting.py --model /path/to/gemma-4-E2B-it.litertlm --backend gpu

  # One-shot prompt generation:
  python3 llmtesting.py --model /path/to/gemma-4-E2B-it.litertlm --backend cpu --prompt "Hi"
"""

import argparse
import glob
import os
import sys

try:
    import litert_lm
    from litert_lm import Backend, Engine
except ImportError:
    print(
        "ERROR: 'litert_lm' is not installed in the current Python environment.\n"
        "Please install it using:\n"
        "  pip install litert-lm\n"
        "or install the pre-release wheel:\n"
        "  pip install <path-to-litert-lm-wheel>.whl\n"
    )
    sys.exit(1)


def find_llm_model(explicit_path: str | None = None) -> str:
    """Finds a .litertlm generative model."""
    if explicit_path:
        expanded = os.path.abspath(os.path.expanduser(explicit_path))
        if os.path.exists(expanded):
            return expanded
        print(f"ERROR: Specified model file does not exist: {expanded}")
        sys.exit(1)

    # Check environment variable
    env_path = os.environ.get("LITERTLM_LLM_MODEL")
    if env_path and os.path.exists(os.path.abspath(os.path.expanduser(env_path))):
        return os.path.abspath(os.path.expanduser(env_path))

    # Check standard local directories
    search_dirs = [
        os.path.join(os.getcwd(), "models"),
        os.getcwd(),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"),
        os.path.dirname(os.path.abspath(__file__)),
    ]

    for s_dir in search_dirs:
        if os.path.exists(s_dir):
            for f in sorted(glob.glob(os.path.join(s_dir, "*.litertlm"))):
                if "embedding" not in os.path.basename(f).lower():
                    return os.path.abspath(f)

    print("ERROR: Could not locate a valid .litertlm LLM model file.")
    print("Please specify the path using '--model <path>':")
    print("  python3 llmtesting.py --model /path/to/model.litertlm --backend cpu")
    sys.exit(1)


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
                print("ℹ️ Note: On macOS, LiteRT-LM uses '--backend gpu' (Apple Metal) for acceleration.")
            sys.exit(1)
    else:
        raise ValueError(f"Unsupported backend '{name}'. Choose 'cpu', 'gpu', or 'npu'.")


def run_interactive_chat(engine: Engine, max_tokens: int = 512) -> None:
    """Runs an interactive multi-turn conversation session in the terminal."""
    print("=" * 70)
    print("  LiteRT-LM Multi-Turn Chat Ready!")
    print("  Type your message below. Type 'exit', 'quit', or press Ctrl+C to stop.")
    print("=" * 70 + "\n")

    with engine.create_conversation(max_output_tokens=max_tokens) as conversation:
        while True:
            try:
                user_input = input("You: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n\nGoodbye!")
                break

            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break

            print("Model: ", end="", flush=True)
            try:
                # Stream the reply token-by-token in real time
                for chunk in conversation.send_message_async(user_input):
                    print(chunk, end="", flush=True)
                print("\n")
            except KeyboardInterrupt:
                print("\n[Generation interrupted]\n")
            except Exception as e:
                print(f"\n[Error generating response: {e}]\n")


def main():
    parser = argparse.ArgumentParser(
        description="LiteRT-LM Interactive Multi-Turn Chat & Testing CLI"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to .litertlm generative LLM model file (e.g. gemma-4-E2B-it.litertlm)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["cpu", "gpu", "npu"],
        default="cpu",
        help="Execution backend: 'cpu', 'gpu', or 'npu' (default: cpu)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="CPU thread count (default: 4)",
    )
    parser.add_argument(
        "--dispatch-lib-dir",
        type=str,
        default=None,
        help="Path to LiteRT dispatch library directory for NPU",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max output tokens generated per turn (default: 512)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Optional single prompt to test non-interactively",
    )

    args = parser.parse_args()

    model_path = find_llm_model(args.model)
    backend_obj = get_backend_instance(
        args.backend, threads=args.threads, dispatch_lib_dir=args.dispatch_lib_dir
    )

    print(f"[+] Loading model: {os.path.basename(model_path)} ({args.backend.upper()})...")

    with Engine(model_path=model_path, backend=backend_obj) as engine:
        if args.prompt:
            # One-shot mode if --prompt is passed
            print(f"You: {args.prompt}")
            print("Model: ", end="", flush=True)
            with engine.create_conversation(max_output_tokens=args.max_tokens) as conv:
                for chunk in conv.send_message_async(args.prompt):
                    print(chunk, end="", flush=True)
                print("\n")
        else:
            # Interactive multi-turn chat
            run_interactive_chat(engine, max_tokens=args.max_tokens)


if __name__ == "__main__":
    main()
