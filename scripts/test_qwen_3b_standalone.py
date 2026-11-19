"""
scripts/test_qwen_3b_standalone.py
─────────────────────────────────────────────────────────────
Standalone inference and memory measurement for qwen2.5:3b.
─────────────────────────────────────────────────────────────
"""

import json
import time
import httpx
import psutil

OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "qwen2.5:3b"

def get_ram_info():
    vm = psutil.virtual_memory()
    return {
        "available_gb": round(vm.available / (1024 ** 3), 2),
        "available_mb": round(vm.available / (1024 ** 2), 1),
        "used_gb": round(vm.used / (1024 ** 3), 2),
        "percent": vm.percent,
    }

def main():
    print("=" * 70)
    print("QWEN 2.5 3B STANDALONE INFERENCE & RESOURCE BENCHMARK")
    print("=" * 70)

    # 1. Baseline RAM
    ram_init = get_ram_info()
    print(f"[BASELINE RAM] Available: {ram_init['available_gb']} GB ({ram_init['available_mb']} MB) | Used: {ram_init['percent']}%")

    with httpx.Client(timeout=120.0) as client:
        # 2. Check installed models
        r_tags = client.get(f"{OLLAMA_URL}/api/tags")
        assert r_tags.status_code == 200, f"Failed to get tags: {r_tags.text}"
        installed_models = [m["name"] for m in r_tags.json().get("models", [])]
        print(f"[INSTALLED MODELS]: {installed_models}")

        # Assert all 3 models are present
        assert any("qwen2.5:3b" in m for m in installed_models), "qwen2.5:3b NOT found in installed models!"
        assert any("llama3" in m for m in installed_models), "llama3:latest NOT found (must be preserved)!"
        assert any("qwen2.5:0.5b" in m for m in installed_models), "qwen2.5:0.5b NOT found (must be preserved)!"
        print("[VERIFICATION] All 3 models verified present on disk.")

        # 3. Check running models before inference
        r_ps_before = client.get(f"{OLLAMA_URL}/api/ps")
        running_before = r_ps_before.json().get("models", [])
        print(f"[OLLAMA_PS_BEFORE] Running models in RAM: {[m.get('name') for m in running_before]}")

        # 4. Run factual inference test
        test_prompt = "What is Retrieval-Augmented Generation (RAG) and what are its two primary stages? Answer concisely in 2-3 sentences."
        print(f"\n[INFERENCE TEST] Prompt: '{test_prompt}'")

        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "user", "content": test_prompt}
            ],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 256,
                "num_ctx": 2048,
            }
        }

        print("[INFERENCE] Sending request to Ollama...")
        t0 = time.perf_counter()
        resp = client.post(f"{OLLAMA_URL}/api/chat", json=payload)
        t1 = time.perf_counter()
        ram_during = get_ram_info()

        assert resp.status_code == 200, f"Inference failed: {resp.text}"
        data = resp.json()
        content = data.get("message", {}).get("content", "").strip()

        # Telemetry
        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)
        prompt_eval_duration_ns = data.get("prompt_eval_duration", 0)
        eval_duration_ns = data.get("eval_duration", 0)
        total_duration_ns = data.get("total_duration", 0)
        load_duration_ns = data.get("load_duration", 0)

        latency_s = t1 - t0
        eval_speed = round(completion_tokens / (eval_duration_ns / 1e9), 2) if eval_duration_ns > 0 else 0.0
        prompt_speed = round(prompt_tokens / (prompt_eval_duration_ns / 1e9), 2) if prompt_eval_duration_ns > 0 else 0.0

        print("\n" + "-" * 70)
        print("GENERATED ANSWER:")
        print(content)
        print("-" * 70)

        # 5. Measure Ollama resident memory via /api/ps
        r_ps_active = client.get(f"{OLLAMA_URL}/api/ps")
        active_models = r_ps_active.json().get("models", [])
        print("\n[OLLAMA /api/ps ACTIVE MEASUREMENTS]:")
        qwen_active_mb = 0
        for m in active_models:
            size_mb = round(m.get("size", 0) / (1024 * 1024), 1)
            size_vram_mb = round(m.get("size_vram", 0) / (1024 * 1024), 1)
            print(f"  Model Name:         {m.get('name')}")
            print(f"  Resident RAM Size:  {size_mb} MB ({round(size_mb/1024, 2)} GB)")
            print(f"  VRAM Allocation:    {size_vram_mb} MB (CPU execution)")
            if "qwen2.5:3b" in m.get("name", ""):
                qwen_active_mb = size_mb

        print(f"\n[HOST RAM DURING INFERENCE]:")
        print(f"  Available RAM:      {ram_during['available_gb']} GB ({ram_during['available_mb']} MB)")
        print(f"  RAM Utilization:    {ram_during['percent']}%")

        print(f"\n[PERFORMANCE TELEMETRY]:")
        print(f"  Prompt Tokens:      {prompt_tokens}")
        print(f"  Completion Tokens:  {completion_tokens}")
        print(f"  Total Tokens:       {prompt_tokens + completion_tokens}")
        print(f"  Prompt Eval Speed:  {prompt_speed} tokens/sec")
        print(f"  Generation Speed:   {eval_speed} tokens/sec")
        print(f"  Cold Load Duration: {round(load_duration_ns / 1e6, 1)} ms")
        print(f"  Total Latency:      {latency_s:.2f} s ({round(total_duration_ns / 1e6, 1)} ms)")

        # 6. Unload model via keep_alive: 0
        print("\n[UNLOAD] Evicting model from RAM via keep_alive: 0...")
        r_unload = client.post(f"{OLLAMA_URL}/api/generate", json={"model": MODEL_NAME, "keep_alive": 0})
        print(f"[UNLOAD] Eviction response status: {r_unload.status_code}")

        # 7. Verify /api/ps has no running models
        time.sleep(1.0)
        r_ps_after = client.get(f"{OLLAMA_URL}/api/ps")
        running_after = r_ps_after.json().get("models", [])
        print(f"[OLLAMA_PS_AFTER] Running models in RAM: {[m.get('name') for m in running_after]}")
        assert len(running_after) == 0, f"Model still in RAM: {running_after}"
        print("[UNLOAD SUCCESS] Model verified 100% evicted from RAM.")

        ram_restored = get_ram_info()
        print(f"\n[FINAL RESTORED HOST RAM]: Available: {ram_restored['available_gb']} GB ({ram_restored['available_mb']} MB) | Used: {ram_restored['percent']}%")
        print("=" * 70)

if __name__ == "__main__":
    main()
