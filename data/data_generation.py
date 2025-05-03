import json
import random

def generate_task_a_data():
    categories = {
        "Camera Scan": [
            "Scanned a Walgreens receipt at 4pm today.",
            "Used the app camera to capture my grocery receipt.",
            "Snapped a picture of my Starbucks bill.",
            "Uploaded a photo of the gas station receipt using the app.",
            "Scanned the receipt from Target manually with my phone."
        ],
        "Email Upload": [
            "Forwarded my Amazon receipt from Gmail.",
            "Uploaded e-receipt from Walmart through email.",
            "Sent digital order confirmation from BestBuy.",
            "Shared Uber Eats receipt from inbox.",
            "Uploaded a Lyft ride receipt via email."
        ],
        "Linked Account": [
            "Synced my Kroger loyalty account to Fetch.",
            "Auto-imported CVS purchases from linked account.",
            "Target receipts show up automatically after syncing.",
            "Connected store card for automatic receipt capture.",
            "Transaction synced through Walgreens membership."
        ],
        "Manual Input": [
            "Typed in my receipt details manually.",
            "Entered items and price line-by-line.",
            "Filled out the receipt form on the app.",
            "Manually logged the products and store name.",
            "Submitted my purchase manually without scanning."
        ]
    }

    data = []
    for label, sentences in categories.items():
        for sentence in sentences:
            data.append({"text": sentence, "label": label})

    random.shuffle(data)

    with open("data/task_a_samples.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"✅ Saved {len(data)} examples to data/task_a_samples.json")

if __name__ == "__main__":
    generate_task_a_data()
