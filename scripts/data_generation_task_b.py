import json
import random

def generate_task_b_data():
    examples = [
        ("App keeps crashing every time I open it!", 0.95),
        ("I lost all my points after the update. Please help.", 0.85),
        ("Scan failed again and again. Frustrating!", 0.9),
        ("Where are my missing rewards? It's been a week!", 0.75),
        ("Can’t log into my account — says wrong password!", 0.8),
        ("App is slow but usable.", 0.4),
        ("I wish the design was cleaner.", 0.3),
        ("How do I change my email address?", 0.2),
        ("Can you resend the verification email?", 0.1),
        ("Loving the new features. Thanks!", 0.0),
    ]

    # Duplicate and shuffle
    data = []
    for _ in range(8):  # 10 x 8 = 80 samples
        for text, score in examples:
            noise = random.uniform(-0.05, 0.05)
            noisy_score = min(max(score + noise, 0.0), 1.0)
            data.append({"text": text, "score": round(noisy_score, 2)})

    random.shuffle(data)

    with open("data/task_b_samples.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"✅ Saved {len(data)} examples to data/task_b_samples.json")

if __name__ == "__main__":
    generate_task_b_data()
