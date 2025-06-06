import os
import json

def save_log(log_dict, save_path="log.json"):
    """학습 로그를 JSON으로 저장."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(log_dict, f, indent=2, ensure_ascii=False)
    print(f"📄 로그 저장 완료: {save_path}")
