# LiteRT-LM Test & Benchmark Suite

A standardized, multi-platform test and benchmark suite for **LiteRT-LM** on-device models, supporting both **Generative LLMs** and **Multimodal Embedding Models**.

Tested across **macOS (Apple Silicon Metal & CPU)**, **Linux (Vulkan, CPU, NPU)**, and **Windows (DirectX/Vulkan, CPU, NPU)**.

---

## 1. Environment Setup

### Step 1: Remove Old LiteRT-LM Packages
Before testing any new build, ensure no conflicting or older wheels are cached:

```bash
pip uninstall -y litert-lm litert-lm-nightly litert_lm_api
```

### Step 2: Create a Clean Virtual Environment

#### On macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

#### On Windows (Command Prompt or PowerShell):
```cmd
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

---

## 2. Installation Options

### Option A: Pre-Release Testing from GCS (Internal Testing Before PyPI)
If you are validating an internal release build before publishing to PyPI, download the wheel from Google Cloud Storage (GCS) or internal release artifacts:

```bash
# Example: Download api wheels from GCS bucket
gcloud storage cp gs://<your-gcs-bucket>/litert_lm_api-*.whl .

# Install the wheel (no other dependencies required!)
pip install litert_lm_api-*.whl
```

### Option B: Post-Publishing Testing (PyPI)
Once the release is published to PyPI, install directly:

```bash
pip install litert-lm
```

---

## 3. Supported Models & Download Links

LiteRT-LM models use the `.litertlm` container format. You can download the official models directly from **Hugging Face Hub** or **Kaggle**:

### 1. Generative LLM Model (`gemma-4-E2B-it.litertlm`)
- **Hugging Face Hub**: [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it)
  - Direct Model Repository: [huggingface.co/google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it)
  - Files & Artifacts: [huggingface.co/google/gemma-4-E2B-it/tree/main](https://huggingface.co/google/gemma-4-E2B-it/tree/main)

### 2. Multimodal Embedding Model (`embedding-gemma-v2.litertlm`)
- **Hugging Face Hub**:
  - [Embedding Gemma on Hugging Face](https://huggingface.co/google/embeddinggemma-2b)
- **Kaggle Models**:
  - [Google Embedding Gemma on Kaggle](https://www.kaggle.com/models/google/embeddinggemma)

---

### Model Placement
Place your downloaded `.litertlm` files in the repository root or in a `./models/` directory:
- `gemma-4-E2B-it.litertlm` (Interactive chat with `llmtesting.py`)
- `embedding-gemma-v2.litertlm` (Embedding verification & benchmarks with `embeddinggemmatesting.py`)

Both testing scripts will **automatically discover** these models without needing `--model`. You can also supply any custom path with `--model <path>`.

---

## 4. Testing Guide by OS & Hardware Backend

Model files (`.litertlm`) are multi-gigabyte binaries and are excluded from Git. Pass the path using `--model` or place them in `./models/`.

---

### A. Multimodal Embedding Testing (`embeddinggemmatesting.py`)

Verifies single embedding latency (Text, Vision, Audio) and cosine similarity ranking (2 Same vs 1 Opposite, and Cross-Modal).

#### 🍎 macOS (Apple Silicon: M1 / M2 / M3 / M4)
- **CPU only**:
  ```bash
  python3 embeddinggemmatesting.py --model embedding-gemma-v2.litertlm --backend cpu
  ```
- **GPU (Apple Metal via WebGPU)**:
  ```bash
  python3 embeddinggemmatesting.py --model embedding-gemma-v2.litertlm --backend gpu
  ```
- **Both CPU & GPU (with Comparative Speedup Table)**:
  ```bash
  python3 embeddinggemmatesting.py --model embedding-gemma-v2.litertlm --backend both
  ```
> **Note on NPU on macOS**: LiteRT-LM leverages Apple Silicon GPU (Metal) for hardware acceleration. Apple Neural Engine (ANE) via NPU backend is not currently exposed in LiteRT-LM. Always test with `--backend gpu` on Apple Silicon.

#### 🐧 Linux (x86_64 / aarch64)
- **CPU only**:
  ```bash
  python3 embeddinggemmatesting.py --model embedding-gemma-v2.litertlm --backend cpu
  ```
- **GPU (Vulkan / WebGPU)**:
  ```bash
  python3 embeddinggemmatesting.py --model embedding-gemma-v2.litertlm --backend gpu
  ```
- **Both CPU & GPU (with Comparative Speedup Table)**:
  ```bash
  python3 embeddinggemmatesting.py --model embedding-gemma-v2.litertlm --backend both
  ```

#### 🪟 Windows (DirectX 12 / Vulkan / CPU / NPU)
- **CPU only**:
  ```cmd
  python embeddinggemmatesting.py --model embedding-gemma-v2.litertlm --backend cpu
  ```
- **GPU (DirectX 12 / Vulkan via WebGPU)**:
  ```cmd
  python embeddinggemmatesting.py --model embedding-gemma-v2.litertlm --backend gpu
  ```
- **Both CPU & GPU**:
  ```cmd
  python embeddinggemmatesting.py --model embedding-gemma-v2.litertlm --backend both
  ```
- **NPU (Windows on ARM / Qualcomm / Intel AI Boost)**:
  ```cmd
  python embeddinggemmatesting.py --model embedding-gemma-v2.litertlm --backend npu
  ```

---

### B. Generative LLM Interactive Chat (`llmtesting.py`)

Launches an interactive multi-turn conversation session directly in the terminal. Type your messages (e.g. `hi`), and the model streams replies in real time with conversational memory.

#### 🍎 macOS (Apple Silicon)
- **CPU**:
  ```bash
  python3 llmtesting.py --model gemma-4-E2B-it.litertlm --backend cpu
  ```
- **GPU (Apple Metal)**:
  ```bash
  python3 llmtesting.py --model gemma-4-E2B-it.litertlm --backend gpu
  ```

#### 🐧 Linux
- **CPU**:
  ```bash
  python3 llmtesting.py --model gemma-4-E2B-it.litertlm --backend cpu
  ```
- **GPU (Vulkan / WebGPU)**:
  ```bash
  python3 llmtesting.py --model gemma-4-E2B-it.litertlm --backend gpu
  ```

#### 🪟 Windows
- **CPU**:
  ```cmd
  python llmtesting.py --model gemma-4-E2B-it.litertlm --backend cpu
  ```
- **GPU (DirectX 12 / Vulkan)**:
  ```cmd
  python llmtesting.py --model gemma-4-E2B-it.litertlm --backend gpu
  ```
- **NPU**:
  ```cmd
  python llmtesting.py --model gemma-4-E2B-it.litertlm --backend npu
  ```

#### 💬 Interactive Terminal Chat Example:
```
======================================================================
  LiteRT-LM Multi-Turn Chat Ready!
  Type your message below. Type 'exit', 'quit', or press Ctrl+C to stop.
======================================================================

You: hi
Model: Hi! How can I help you today? 😊

You: What is 2 + 2?
Model: 2 + 2 is **4**.

You: exit
Goodbye!
```

---

## 5. Sample Benchmark Output for Embedding Gemma 2

```
================================================================================
  HARDWARE COMPARISON: CPU vs GPU (Apple Silicon Metal / WebGPU)
================================================================================
Modality   | CPU Latency    | GPU Latency    | Speedup     
--------------------------------------------------------------------------------
Text       |      33.81 ms   |      13.52 ms   |     2.50x faster
Vision     |     379.03 ms   |     131.18 ms   |     2.89x faster
Audio      |     106.41 ms   |      54.22 ms   |     1.96x faster
--------------------------------------------------------------------------------

✅ ALL 4 VERIFICATION TESTS PASSED (Text, Vision, Audio, Cross-Modal)!
```

---

## 6. CLI Arguments Quick Reference

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--model` | Auto-detect | Path to `.litertlm` model file |
| `--backend` | `both` | Backend: `cpu`, `gpu`, `npu`, or `both` |
| `--threads` | `4` | Number of CPU threads |
| `--dispatch-lib-dir` | `None` | Path to LiteRT dispatch library directory (for NPU) |
| `--iterations` | `3` | Number of benchmark runs per backend |
| `--max-tokens` | `60` | Max tokens generated during benchmark (`llmtesting.py`) |
| `--prompt` | Built-in | Custom prompt text (`llmtesting.py`) |
| `--samples-dir` | Auto-detect | Directory containing `vision/` and `audio/` test files (`embeddinggemmatesting.py`) |

---

## 7. Pass / Fail Exit Codes for CI/CD

Both scripts return standard exit codes:
- **Code 0**: All verification tests and benchmarks completed successfully (`PASS`).
- **Code 1**: One or more functional tests failed or error occurred (`FAIL`).
