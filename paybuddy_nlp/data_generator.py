"""
Generates a labelled pilot dataset for the PayBuddy Categorisation Engine /
NLP course project.

Three buckets are generated per category, matching the three classifier
stages, so evaluation can report which stage handles which kind of input:

  bucket "dictionary" -> known merchant, should be caught by Stage 1
  bucket "regex"      -> unseen merchant, but a tell-tale keyword phrase
                         present -> should be caught by Stage 2
  bucket "nlp_hard"   -> unseen merchant, category expressed indirectly /
                         semantically, no dictionary or regex trigger
                         -> genuinely requires Stage 3 (trained NLP model)

Every row also gets light template variety and a little textual noise
(typos, casing) so the task isn't trivially clean.
"""

import random
import re
import pandas as pd

random.seed(42)

TEMPLATES = [
    "POS purchase at {m}",
    "UPI/{m_nospace}/{cat}",
    "Auto-debit - {m}",
    "Payment to {m} via UPI",
    "NEFT credit from {m}",
    "IMPS transfer - {m}",
    "{m}",
]

# ---------------------------------------------------------------------------
# Bucket 1: known merchants (mirrors merchant_dictionary.py exactly)
# ---------------------------------------------------------------------------
KNOWN_MERCHANTS = {
    "Bills": ["Airtel Postpaid", "BESCOM Electricity", "Hostel Rent",
              "Jio Fiber Broadband", "Netflix Subscription"],
    "Education": ["Coursera", "Exam Registration Fee", "Udemy",
                  "University Bookstore"],
    "Entertainment": ["BookMyShow", "Gaming Zone Arcade", "INOX Movies"],
    "Food": ["Campus Cafeteria", "Chai Point", "Domino's Pizza",
             "Local Tiffin Service", "McDonald's", "Swiggy",
             "Third Wave Coffee", "Zomato"],
    "Income": ["Freelance Payment - Upwork", "Monthly Allowance - Parents",
               "Scholarship Disbursement"],
    "Medical": ["1mg Online Pharmacy", "Apollo Pharmacy", "Practo Consultation"],
    "Other": ["ATM Cash Withdrawal", "Bank Charges", "Miscellaneous POS",
              "UPI Transfer - Friend", "Unidentified Merchant"],
    "Shopping": ["Amazon India", "Croma", "Decathlon", "Flipkart",
                 "Local Bookstore", "Reliance Trends"],
    "Travel": ["BMTC Bus Pass", "IndiGo Airlines", "Ola Cabs", "RedBus", "Uber"],
}

# ---------------------------------------------------------------------------
# Bucket 2: unseen merchants with a regex-catchable keyword phrase baked in
# ---------------------------------------------------------------------------
REGEX_MERCHANTS = {
    "Bills": ["Vodafone Idea electricity bill payment", "Home broadband renewal",
              "DTH recharge - Tata Play", "Monthly wifi bill - ACT Fibernet",
              "PG rent due for October", "Society maintenance charge"],
    "Education": ["Skill Lync course fee", "GATE exam fee payment",
                  "Central Library fine cleared", "Workshop fee - IEEE chapter",
                  "College semester fee installment"],
    "Entertainment": ["PVR movie ticket booking", "Fun World amusement park",
                      "Local band concert ticket", "Timezone gaming zone",
                      "Spotify OTT subscription renewal"],
    "Food": ["Rajesh Dhaba dinner", "Corner cafe evening snack",
             "New Bakery order", "Hostel mess food payment",
             "Swasthi restaurant bill", "Faasos food delivery order"],
    "Medical": ["City Care pharmacy purchase", "St Johns hospital bill",
                "Skin clinic consultation", "Metro medical store purchase",
                "Thyrocare lab test payment"],
    "Shopping": ["Metro supermarket bill", "Big Bazaar department store",
                 "Vijay electronics store purchase", "Trendz clothing store",
                 "City Centre shopping mall purchase"],
    "Travel": ["Rapido cab ride", "Meru taxi booking", "IndiGo flight ticket",
               "KSRTC bus ticket booking", "Namma Metro card recharge",
               "HP petrol pump fuel fill", "IRCTC train ticket booking"],
    "Income": ["Infosys salary credited", "TCS stipend credited",
               "Amazon refund received", "HDFC cashback credited",
               "Hackathon prize money transfer"],
    "Other": ["ATM withdrawal at HDFC", "SBI bank charge deducted",
              "Misc UPI transfer to unknown", "Petty transfer to roommate"],
}

# ---------------------------------------------------------------------------
# Bucket 3: unseen merchants, category expressed indirectly (no dict/regex hit)
# These are free-text-style descriptions, closer to how a person might
# describe a spend rather than a bank's auto-generated line.
# ---------------------------------------------------------------------------
NLP_HARD_EXAMPLES = {
    "Bills": [
        "cleared this months electricity due", "wifi connection renewed for the flat",
        "warden collected the room deposit this week", "annual maintenance charge for the apartment",
        "topped up the mobile network balance", "paid the landlord for this month",
    ],
    "Education": [
        "seminar registration completed online", "id card renewal charge at admin office",
        "bought lab manual for the semester", "paid for the workshop materials",
        "cleared pending dues at the department office", "certification exam booked for next month",
    ],
    "Entertainment": [
        "weekend outing with friends at the arcade", "tickets booked for the concert downtown",
        "subscribed to a music streaming service", "spent the evening at a rooftop lounge",
        "booked seats for the comedy show", "paid entry fee for the college fest",
    ],
    "Food": [
        "grabbed a quick bite between classes", "shared a meal with roommates at the mess",
        "ordered dessert after dinner", "bought snacks from the tuck shop",
        "treated myself to a milkshake", "paid for lunch with the project team",
    ],
    "Income": [
        "amount received from a part time gig", "parents sent money for the month",
        "got paid for the tutoring session", "prize money for winning the hackathon",
        "reimbursement credited from college", "sold old textbooks to a junior",
    ],
    "Medical": [
        "consultation with the family doctor", "picked up prescribed tablets today",
        "annual checkup at the clinic", "eye test at the optician",
        "dental appointment payment", "bought a thermometer and bandages",
    ],
    "Other": [
        "sent money to a friend for splitting rent", "withdrew cash from the machine near campus",
        "small charge deducted by the bank", "paid back a friend for lending cash",
        "unclear charge on the statement", "donated a small amount to a fundraiser",
    ],
    "Shopping": [
        "picked up a few things from the general store", "bought a new pair of shoes",
        "ordered a gift for a friend online", "new charger for the laptop",
        "bought a study table lamp", "picked up a birthday present for a friend",
    ],
    "Travel": [
        "booked a ride back home for the weekend", "train ticket for the trip next month",
        "flight booked for the semester break", "shared an auto to the railway station",
        "booked a cab to the airport early morning", "topped up the travel card",
    ],
}

TYPO_SWAPS = [("a", "@"), ("e", "3"), ("purchase", "purchse"), ("payment", "paymnt")]


def _apply_light_noise(text, p=0.12):
    if random.random() < p:
        old, new = random.choice(TYPO_SWAPS)
        if old in text:
            text = text.replace(old, new, 1)
    if random.random() < 0.15:
        text = text.lower()
    return text


NON_MERCHANT_TEMPLATES = [t for t in TEMPLATES if "{m_nospace}" not in t]


def _make_row(merchant_or_phrase, category, bucket, use_template=True):
    if use_template:
        # the compressed 'UPI/{merchant}/{category}' style reference is only
        # realistic for an actual short merchant name, not a full descriptive
        # phrase -> restrict it to the dictionary bucket.
        pool = TEMPLATES if bucket == "dictionary" else NON_MERCHANT_TEMPLATES
        template = random.choice(pool)
        desc = template.format(
            m=merchant_or_phrase,
            m_nospace=re.sub(r"[^A-Za-z0-9]", "", merchant_or_phrase),
            cat=category,
        )
    else:
        desc = merchant_or_phrase.capitalize()
    desc = _apply_light_noise(desc)
    return {
        "description": desc,
        "category": category,
        "bucket": bucket,
    }


def generate_dataset(n_per_merchant_dict=4, n_per_regex_phrase=3, n_per_hard_phrase=3):
    rows = []

    for category, merchants in KNOWN_MERCHANTS.items():
        for m in merchants:
            for _ in range(n_per_merchant_dict):
                rows.append(_make_row(m, category, "dictionary"))

    for category, phrases in REGEX_MERCHANTS.items():
        for phrase in phrases:
            for _ in range(n_per_regex_phrase):
                rows.append(_make_row(phrase, category, "regex", use_template=random.random() > 0.4))

    for category, phrases in NLP_HARD_EXAMPLES.items():
        for phrase in phrases:
            for _ in range(n_per_hard_phrase):
                rows.append(_make_row(phrase, category, "nlp_hard", use_template=False))

    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    df.insert(0, "transaction_id", range(1, len(df) + 1))
    return df


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("pilot_dataset.csv", index=False)
    print(df.shape)
    print(df["bucket"].value_counts())
    print(df["category"].value_counts())
