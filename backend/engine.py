import json
import glob
import numpy as np

def decision_engine(all_results):
    suspicious_cnt, negemo_cnt, gaze_drift_cnt, pause_cnt = 0,0,0,0

    for r in all_results:
        ans = r["analysis"]

        if ans["suspicious"]:
            suspicious_cnt += 1

        # Эмоции
        emos = [e["emotion"] for e in ans["emotions"]]
        if emos.count("fear") > 2 or emos.count("angry") > 2:
            negemo_cnt += 1

        # Взгляд
        gazes = ans["gaze"]
        if len(gazes) > 5:
            xs = [g["gaze_x"] for g in gazes]
            if np.std(xs) > 0.05:  # сильный бег глаз = "читает"
                gaze_drift_cnt += 1

        # Паузы
        if len(ans["pauses"]) > 0:
            pause_cnt += 1

    total = len(all_results)
    summary = {
        "questions": total,
        "suspicious_answers": suspicious_cnt,
        "negative_emotional_responses": negemo_cnt,
        "gaze_drift_cases": gaze_drift_cnt,
        "long_pauses_detected": pause_cnt
    }

    if suspicious_cnt/total > 0.3 or gaze_drift_cnt > 1 or negemo_cnt > 1 or pause_cnt > 1:
        summary["final_decision"] = "FAIL"
        summary["reason"] = "слишком много признаков волнения/чтения/заминок"
    else:
        summary["final_decision"] = "PASS"
        summary["reason"] = "отвечал уверенно, без значительных проблем"

    return summary


def main():
    results = []
    for path in glob.glob("*/result.json"):
        with open(path, encoding="utf-8") as f:
            results.append(json.load(f))

    decision = decision_engine(results)

    with open("final_summary.json", "w", encoding="utf-8") as f:
        json.dump(decision, f, ensure_ascii=False, indent=2)

    print("ФИНАЛЬНЫЙ ВЕРДИКТ:")
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
