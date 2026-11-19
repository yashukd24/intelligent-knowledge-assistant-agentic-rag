import json
import shutil
from pathlib import Path

transcript_path = Path(r"C:\Users\YASHWANTH D\.gemini\antigravity\brain\e205a237-382f-4d1d-ab05-e1e628778c65\.system_generated\logs\transcript.jsonl")
media_dir = Path(r"C:\Users\YASHWANTH D\.gemini\antigravity\brain\e205a237-382f-4d1d-ab05-e1e628778c65\.system_generated\logs\media")
dest_dir = Path("demo_output/screenshots")
dest_dir.mkdir(parents=True, exist_ok=True)

steps_map = {}
with open(transcript_path, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        step = data.get("step_index")
        t_calls = data.get("tool_calls", [])
        media = data.get("media", [])
        if t_calls:
            for tc in t_calls:
                intent = tc.get("args", {}).get("intention", "")
                if intent:
                    print(f"Step {step} intent: {intent}")
        if media:
            for m in media:
                uri = m.get("uri", "")
                fname = uri.split("/")[-1]
                steps_map[step] = fname
                print(f"  Step {step} saved: {fname}")

# Specific mapping based on tool calls and intentions:
# Step 1 intended dashboard screenshot -> media in step 2: step_2_media_0.png
# Step 5 intended collections screenshot -> media in step 6: step_6_media_0.png
# Step 9 intended collection detail screenshot -> media in step 10: step_10_media_0.png
# Step 13 intended chat initial view -> media in step 14: step_14_media_0.png
# Step 24 intended grounded answer -> media in step 25: step_25_media_0.png
# Step 34 intended source drawer -> media in step 35: step_35_media_0.png
# Step 36 intended trace drawer -> media in step 37: step_37_media_0.png
# Step 43 intended unanswerable query refusal -> media in step 44: step_44_media_0.png
# Step 45 intended faculty role -> media in step 46: step_46_media_0.png

target_files = {
    "01_dashboard.png": "step_2_media_0.png",
    "02_collections_list.png": "step_6_media_0.png",
    "03_collection_detail_college.png": "step_10_media_0.png",
    "04_chat_initial_college.png": "step_14_media_0.png",
    "05_chat_cse_hod_grounded.png": "step_25_media_0.png",
    "06_chat_source_drawer.png": "step_35_media_0.png",
    "07_chat_trace_drawer.png": "step_37_media_0.png",
    "08_chat_unanswerable_refusal.png": "step_44_media_0.png",
    "09_chat_faculty_role.png": "step_46_media_0.png",
}

for dest_name, src_name in target_files.items():
    src_file = media_dir / src_name
    if src_file.exists():
        dest_file = dest_dir / dest_name
        shutil.copy2(src_file, dest_file)
        print(f"Copied {src_name} -> {dest_file} ({src_file.stat().st_size} bytes)")
    else:
        print(f"Warning: {src_file} does not exist!")
