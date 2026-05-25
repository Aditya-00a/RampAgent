"""Generate synthetic corporate expense data with planted anomalies."""

import csv
import random
from datetime import datetime, timedelta

random.seed(42)

EMPLOYEES = [
    ("Sarah Chen", "Engineering"),
    ("Marcus Johnson", "Sales"),
    ("Priya Patel", "Marketing"),
    ("David Kim", "Finance"),
    ("Rachel Torres", "Operations"),
    ("James O'Brien", "Engineering"),
    ("Aisha Mohammed", "Product"),
    ("Tom Brennan", "Sales"),
    ("Lisa Wang", "Engineering"),
    ("Carlos Rivera", "Marketing"),
    ("Emily Zhang", "HR"),
    ("Mike Sullivan", "Sales"),
    ("Nina Kowalski", "Product"),
    ("Alex Petrov", "Engineering"),
    ("Jordan Lee", "Operations"),
    ("Hannah Morris", "Finance"),
    ("Ryan Nakamura", "Engineering"),
    ("Sofia Gutierrez", "Marketing"),
    ("Daniel Park", "Legal"),
    ("Megan Foster", "Sales"),
]

NORMAL_VENDORS = {
    "Meals & Dining": [
        ("Sweetgreen", 12, 22), ("Chipotle", 10, 18), ("Blue Bottle Coffee", 5, 12),
        ("Shake Shack", 12, 20), ("Dig Inn", 11, 19), ("Just Salad", 10, 16),
        ("Gramercy Tavern", 45, 75), ("The Smith", 30, 55), ("Dos Toros", 11, 17),
    ],
    "Travel - Flights": [
        ("United Airlines", 250, 480), ("Delta Air Lines", 280, 500),
        ("JetBlue", 180, 380), ("American Airlines", 260, 490),
    ],
    "Travel - Hotels": [
        ("Marriott", 150, 245), ("Hilton", 140, 240), ("Hyatt", 160, 250),
        ("Hampton Inn", 110, 180),
    ],
    "Travel - Ground Transport": [
        ("Uber", 12, 45), ("Lyft", 10, 40), ("Via", 8, 15),
        ("NYC Taxi", 15, 50),
    ],
    "Software & SaaS": [
        ("Figma", 15, 45), ("Notion", 10, 20), ("Slack", 8, 15),
        ("GitHub", 20, 44), ("AWS", 50, 190), ("Zoom", 15, 25),
        ("Linear", 8, 10), ("Vercel", 20, 40),
    ],
    "Office Supplies": [
        ("Staples", 15, 80), ("Amazon Business", 10, 90),
        ("Apple Store", 30, 150), ("Best Buy", 25, 120),
    ],
    "Entertainment": [
        ("StubHub", 40, 95), ("Broadway.com", 60, 100),
        ("Topgolf", 30, 70), ("Bowlero", 25, 60),
    ],
    "Professional Services": [
        ("WeWork", 200, 400), ("Kinkos/FedEx", 15, 60),
        ("DocuSign", 25, 50), ("LegalZoom", 30, 80),
    ],
    "Telecommunications": [
        ("Verizon", 40, 85), ("AT&T", 35, 80), ("T-Mobile", 30, 70),
    ],
    "Training & Education": [
        ("Coursera", 40, 79), ("Udemy", 12, 30), ("O'Reilly Media", 40, 50),
        ("LinkedIn Learning", 30, 40),
    ],
}

DESCRIPTIONS = {
    "Meals & Dining": ["Team lunch", "Solo working lunch", "Client dinner", "Coffee meeting", "Team dinner", "Lunch with candidate"],
    "Travel - Flights": ["Flight to SFO for client meeting", "Flight to LAX for conference", "Flight to ORD for team offsite", "Flight to BOS for partner meeting"],
    "Travel - Hotels": ["2 nights for SFO trip", "3 nights for conference", "1 night for client visit", "2 nights for offsite"],
    "Travel - Ground Transport": ["Uber to airport", "Lyft to client office", "Taxi from JFK", "Ride to dinner venue"],
    "Software & SaaS": ["Monthly subscription", "Annual license renewal", "Team plan upgrade", "Pro tier subscription"],
    "Office Supplies": ["Monitor stand + cables", "Notebook and pens", "USB-C hub", "Desk lamp"],
    "Entertainment": ["Client entertainment - basketball game", "Team outing", "Client dinner event", "Holiday party venue"],
    "Professional Services": ["Coworking day pass", "Print and bind report", "Contract signing service", "Notary service"],
    "Telecommunications": ["Monthly phone plan", "International calling add-on", "Hotspot plan"],
    "Training & Education": ["Online course enrollment", "Certification exam", "Technical workshop", "Annual subscription"],
}


def gen_date(start, end):
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))


def gen_normal_transaction(txn_id, start_date, end_date):
    emp, dept = random.choice(EMPLOYEES)
    cat = random.choice(list(NORMAL_VENDORS.keys()))
    vendor_name, lo, hi = random.choice(NORMAL_VENDORS[cat])
    amount = round(random.uniform(lo, hi), 2)
    date = gen_date(start_date, end_date)
    while date.weekday() >= 5:
        date = gen_date(start_date, end_date)
    hour = random.randint(8, 19)
    date = date.replace(hour=hour, minute=random.randint(0, 59))
    desc = random.choice(DESCRIPTIONS[cat])
    receipt = random.random() < 0.92

    return {
        "transaction_id": f"TXN-{txn_id:04d}",
        "date": date.strftime("%Y-%m-%d %H:%M"),
        "employee_name": emp,
        "department": dept,
        "vendor": vendor_name,
        "category": cat,
        "amount": amount,
        "description": desc,
        "receipt_attached": receipt,
        "pre_approved": cat in ("Travel - Flights", "Travel - Hotels") or random.random() < 0.1,
    }


def plant_anomalies(transactions, start_date, end_date):
    anomalies = []
    txn_id = len(transactions) + 1

    # 1. Over-limit meals ($120-$400)
    for _ in range(10):
        emp, dept = random.choice(EMPLOYEES)
        vendors = [("Nobu", 280, 400), ("Peter Luger", 200, 350), ("Le Bernardin", 250, 380),
                    ("Carbone", 180, 320), ("Eleven Madison Park", 300, 420)]
        v, lo, hi = random.choice(vendors)
        date = gen_date(start_date, end_date)
        hour = random.randint(18, 22)
        date = date.replace(hour=hour, minute=random.randint(0, 59))
        anomalies.append({
            "transaction_id": f"TXN-{txn_id:04d}",
            "date": date.strftime("%Y-%m-%d %H:%M"),
            "employee_name": emp,
            "department": dept,
            "vendor": v,
            "category": "Meals & Dining",
            "amount": round(random.uniform(lo, hi), 2),
            "description": random.choice(["Client dinner", "Team celebration dinner", "Dinner with prospects"]),
            "receipt_attached": random.random() < 0.7,
            "pre_approved": False,
        })
        txn_id += 1

    # 2. Duplicates (same vendor, similar amount, close dates)
    for _ in range(5):
        base = random.choice(transactions)
        dup_date = datetime.strptime(base["date"], "%Y-%m-%d %H:%M") + timedelta(hours=random.randint(1, 36))
        anomalies.append({
            "transaction_id": f"TXN-{txn_id:04d}",
            "date": dup_date.strftime("%Y-%m-%d %H:%M"),
            "employee_name": base["employee_name"],
            "department": base["department"],
            "vendor": base["vendor"],
            "category": base["category"],
            "amount": round(base["amount"] * random.uniform(0.97, 1.03), 2),
            "description": base["description"],
            "receipt_attached": base["receipt_attached"],
            "pre_approved": base["pre_approved"],
        })
        txn_id += 1

    # 3. Weekend entertainment
    for _ in range(5):
        emp, dept = random.choice(EMPLOYEES)
        date = gen_date(start_date, end_date)
        while date.weekday() < 5:
            date = gen_date(start_date, end_date)
        hour = random.randint(19, 23)
        date = date.replace(hour=hour, minute=random.randint(0, 59))
        anomalies.append({
            "transaction_id": f"TXN-{txn_id:04d}",
            "date": date.strftime("%Y-%m-%d %H:%M"),
            "employee_name": emp,
            "department": dept,
            "vendor": random.choice(["StubHub", "Bowlero", "Dave & Buster's", "AMC Theatres", "TopGolf"]),
            "category": "Entertainment",
            "amount": round(random.uniform(120, 350), 2),
            "description": random.choice(["Weekend team event", "Personal entertainment", "Social outing"]),
            "receipt_attached": random.random() < 0.5,
            "pre_approved": False,
        })
        txn_id += 1

    # 4. First-time unusual vendors
    unusual_vendors = [
        ("CryptoGear.io", "Office Supplies", 150, 400),
        ("LuxuryPens.com", "Office Supplies", 200, 500),
        ("PrivateJetShare", "Travel - Ground Transport", 800, 2500),
        ("WineClubDirect", "Entertainment", 200, 600),
        ("DesignerBags.co", "Office Supplies", 300, 900),
        ("ExoticCars Rental", "Travel - Ground Transport", 400, 1200),
        ("SpaRetreat NYC", "Professional Services", 250, 500),
        ("PersonalTrainerPro", "Training & Education", 150, 400),
    ]
    for v, cat, lo, hi in unusual_vendors:
        emp, dept = random.choice(EMPLOYEES)
        date = gen_date(start_date, end_date)
        date = date.replace(hour=random.randint(10, 17), minute=random.randint(0, 59))
        anomalies.append({
            "transaction_id": f"TXN-{txn_id:04d}",
            "date": date.strftime("%Y-%m-%d %H:%M"),
            "employee_name": emp,
            "department": dept,
            "vendor": v,
            "category": cat,
            "amount": round(random.uniform(lo, hi), 2),
            "description": "Purchase",
            "receipt_attached": random.random() < 0.4,
            "pre_approved": False,
        })
        txn_id += 1

    # 5. Late-night transactions
    for _ in range(5):
        emp, dept = random.choice(EMPLOYEES)
        date = gen_date(start_date, end_date)
        hour = random.choice([0, 1, 2, 3, 4, 23])
        date = date.replace(hour=hour, minute=random.randint(0, 59))
        anomalies.append({
            "transaction_id": f"TXN-{txn_id:04d}",
            "date": date.strftime("%Y-%m-%d %H:%M"),
            "employee_name": emp,
            "department": dept,
            "vendor": random.choice(["Uber", "DoorDash", "GrubHub", "Bar Tab NYC"]),
            "category": random.choice(["Meals & Dining", "Travel - Ground Transport"]),
            "amount": round(random.uniform(40, 180), 2),
            "description": random.choice(["Late night ride", "Late night food delivery", "After-hours transport"]),
            "receipt_attached": random.random() < 0.3,
            "pre_approved": False,
        })
        txn_id += 1

    # 6. Missing receipts on high-value items
    for _ in range(10):
        emp, dept = random.choice(EMPLOYEES)
        cat = random.choice(["Software & SaaS", "Professional Services", "Office Supplies"])
        date = gen_date(start_date, end_date)
        date = date.replace(hour=random.randint(9, 17), minute=random.randint(0, 59))
        anomalies.append({
            "transaction_id": f"TXN-{txn_id:04d}",
            "date": date.strftime("%Y-%m-%d %H:%M"),
            "employee_name": emp,
            "department": dept,
            "vendor": random.choice(["Amazon Business", "Best Buy", "B&H Photo", "Micro Center"]),
            "category": cat,
            "amount": round(random.uniform(150, 500), 2),
            "description": random.choice(["Equipment purchase", "Hardware upgrade", "Tech supplies"]),
            "receipt_attached": False,
            "pre_approved": False,
        })
        txn_id += 1

    # 7. Velocity spikes (one employee, many transactions in a few days)
    spike_emp, spike_dept = EMPLOYEES[3]  # David Kim, Finance
    for i in range(8):
        date = start_date + timedelta(days=45 + i // 3)
        date = date.replace(hour=random.randint(9, 18), minute=random.randint(0, 59))
        anomalies.append({
            "transaction_id": f"TXN-{txn_id:04d}",
            "date": date.strftime("%Y-%m-%d %H:%M"),
            "employee_name": spike_emp,
            "department": spike_dept,
            "vendor": random.choice(["Amazon Business", "Best Buy", "Apple Store", "Staples"]),
            "category": random.choice(["Office Supplies", "Software & SaaS"]),
            "amount": round(random.uniform(80, 350), 2),
            "description": random.choice(["Bulk purchase", "Equipment order", "Rush order"]),
            "receipt_attached": random.random() < 0.6,
            "pre_approved": False,
        })
        txn_id += 1

    # 8. Software over $200 without IT approval
    for _ in range(7):
        emp, dept = random.choice([e for e in EMPLOYEES if e[1] != "Engineering"])
        date = gen_date(start_date, end_date)
        date = date.replace(hour=random.randint(9, 17), minute=random.randint(0, 59))
        anomalies.append({
            "transaction_id": f"TXN-{txn_id:04d}",
            "date": date.strftime("%Y-%m-%d %H:%M"),
            "employee_name": emp,
            "department": dept,
            "vendor": random.choice(["Adobe Creative Cloud", "Salesforce", "HubSpot", "Atlassian", "Datadog", "Snowflake"]),
            "category": "Software & SaaS",
            "amount": round(random.uniform(210, 800), 2),
            "description": random.choice(["Enterprise license", "Annual plan upgrade", "Premium tier"]),
            "receipt_attached": True,
            "pre_approved": False,
        })
        txn_id += 1

    # 9. Flights without pre-approval (over $500)
    for _ in range(5):
        emp, dept = random.choice(EMPLOYEES)
        date = gen_date(start_date, end_date)
        date = date.replace(hour=random.randint(6, 20), minute=random.randint(0, 59))
        anomalies.append({
            "transaction_id": f"TXN-{txn_id:04d}",
            "date": date.strftime("%Y-%m-%d %H:%M"),
            "employee_name": emp,
            "department": dept,
            "vendor": random.choice(["United Airlines", "Delta Air Lines", "American Airlines"]),
            "category": "Travel - Flights",
            "amount": round(random.uniform(520, 1200), 2),
            "description": random.choice(["Last-minute flight booking", "Business class upgrade", "International flight"]),
            "receipt_attached": True,
            "pre_approved": False,
        })
        txn_id += 1

    return anomalies


if __name__ == "__main__":
    start_date = datetime(2024, 1, 2)
    end_date = datetime(2024, 3, 29)

    transactions = []
    for i in range(1, 433):
        transactions.append(gen_normal_transaction(i, start_date, end_date))

    anomalies = plant_anomalies(transactions, start_date, end_date)
    all_txns = transactions + anomalies
    random.shuffle(all_txns)

    # Re-assign sequential IDs after shuffle
    for i, t in enumerate(all_txns):
        t["transaction_id"] = f"TXN-{i + 1:04d}"

    with open("C:/Ravendise/RampAgent/data/sample_transactions.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "transaction_id", "date", "employee_name", "department",
            "vendor", "category", "amount", "description",
            "receipt_attached", "pre_approved",
        ])
        writer.writeheader()
        writer.writerows(all_txns)

    print(f"Generated {len(all_txns)} transactions ({len(transactions)} normal + {len(anomalies)} anomalies)")
