import json
import random

def generate_task_a_data():
    categories = {
        "Camera Scan": [
            "Scanned a Walgreens receipt at 4pm today.",
            "Used the app camera to capture my grocery receipt.",
            "Snapped a picture of my Starbucks bill.",
            "Uploaded a photo of the gas station receipt using the app.",
            "Scanned the receipt from Target manually with my phone.",
            "Captured the McDonald’s receipt using the app camera.",
            "Used the mobile scan feature to upload my receipt.",
            "Submitted a photo of my lunch receipt.",
            "Scanned and uploaded a paper bill from Kroger.",
            "Snapped and submitted a receipt from 7-Eleven.",
            "Took a picture of the CVS bill using my phone.",
            "Scanned a printed bill and uploaded it.",
            "Used in-app camera to scan my Whole Foods receipt.",
            "Submitted a high-res photo of the store receipt.",
            "Used mobile camera to upload my Trader Joe’s receipt.",
            "Scanned the grocery bill using the app.",
            "Took a snapshot of the printed invoice.",
            "Snapped the bill with Fetch's scan option.",
            "Used scan feature to record the fuel receipt.",
            "Photographed a Target receipt with the app."
        ],
        "Email Upload": [
            "Forwarded my Amazon receipt from Gmail.",
            "Uploaded e-receipt from Walmart through email.",
            "Sent digital order confirmation from BestBuy.",
            "Shared Uber Eats receipt from inbox.",
            "Uploaded a Lyft ride receipt via email.",
            "Forwarded my Grubhub order confirmation.",
            "Submitted DoorDash receipt from email.",
            "Mailed in a receipt from Instacart.",
            "Sent over a digital purchase receipt.",
            "Uploaded my Apple order confirmation from email.",
            "Gmail receipt from Domino’s was uploaded.",
            "Emailed a flight booking receipt to Fetch.",
            "Shared Uber receipt from Outlook inbox.",
            "Sent in a hotel booking receipt from Gmail.",
            "Submitted Airbnb payment receipt via email.",
            "Shared Gmail copy of my recent Starbucks order.",
            "Sent Walgreens digital invoice to Fetch email.",
            "Forwarded a concert ticket receipt.",
            "Uploaded movie ticket e-receipt from email.",
            "Sent invoice for Amazon Fresh via email."
        ],
        "Linked Account": [
            "Synced my Kroger loyalty account to Fetch.",
            "Auto-imported CVS purchases from linked account.",
            "Target receipts show up automatically after syncing.",
            "Connected store card for automatic receipt capture.",
            "Transaction synced through Walgreens membership.",
            "Linked my RiteAid rewards account.",
            "Costco transactions are imported via account link.",
            "Publix purchases appear from my linked card.",
            "Sam’s Club receipt showed up via sync.",
            "My digital Target account pulls in receipts.",
            "Fetch auto-syncs with my Meijer account.",
            "Walgreens digital loyalty card is connected.",
            "Synced up Whole Foods digital account.",
            "My Amazon orders sync via account link.",
            "Synced BJ’s membership for receipt imports.",
            "Auto-imported recent orders from linked CVS account.",
            "Fetch pulled receipts from my Safeway account.",
            "Transactions from Giant Eagle are auto-added.",
            "BestBuy purchases synced via digital account.",
            "Linked my Fetch profile with my Kroger Plus card."
        ],
        "Manual Input": [
            "Typed in my receipt details manually.",
            "Entered items and price line-by-line.",
            "Filled out the receipt form on the app.",
            "Manually logged the products and store name.",
            "Submitted my purchase manually without scanning.",
            "I input each item by hand instead of uploading a photo.",
            "Wrote down the details from my printed receipt.",
            "Didn't use camera or email — added everything manually.",
            "Used the 'manual entry' section in the app.",
            "Logged my Starbucks receipt manually.",
            "I typed each item and the total manually.",
            "Manually filled the fields instead of scanning.",
            "Entered the store and total amount by myself.",
            "Didn't upload anything, just wrote it all in.",
            "No scan or sync — just manual receipt logging.",
            "Manually recorded my gas station purchase.",
            "Manually typed my restaurant order into the app.",
            "Inputted details from a handwritten bill.",
            "Submitted itemized entries manually into Fetch.",
            "Entered line items and totals by hand."
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
