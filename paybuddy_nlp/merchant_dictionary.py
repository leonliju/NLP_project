"""
Stage 1 — Merchant Dictionary
Exact-match lookup: normalised merchant name substring -> category.
This represents merchants PayBuddy has already "seen" (from the existing
college.duckdb transaction history). Confidence for a dictionary hit = 0.95.
"""

# key: normalised (lowercase, no punctuation) merchant token(s) that must
# appear as a substring in the normalised description.
# value: ground-truth category label.
MERCHANT_DICTIONARY = {
    # Bills
    "airtel postpaid": "Bills",
    "bescom electricity": "Bills",
    "hostel rent": "Bills",
    "jio fiber broadband": "Bills",
    "netflix subscription": "Bills",
    # Education
    "coursera": "Education",
    "exam registration fee": "Education",
    "udemy": "Education",
    "university bookstore": "Education",
    # Entertainment
    "bookmyshow": "Entertainment",
    "gaming zone arcade": "Entertainment",
    "inox movies": "Entertainment",
    # Food
    "campus cafeteria": "Food",
    "chai point": "Food",
    "dominos pizza": "Food",
    "local tiffin service": "Food",
    "mcdonalds": "Food",
    "swiggy": "Food",
    "third wave coffee": "Food",
    "zomato": "Food",
    # Income
    "freelance payment upwork": "Income",
    "monthly allowance parents": "Income",
    "scholarship disbursement": "Income",
    # Medical
    "1mg online pharmacy": "Medical",
    "apollo pharmacy": "Medical",
    "practo consultation": "Medical",
    # Other
    "atm cash withdrawal": "Other",
    "bank charges": "Other",
    "miscellaneous pos": "Other",
    "upi transfer friend": "Other",
    "unidentified merchant": "Other",
    # Shopping
    "amazon india": "Shopping",
    "croma": "Shopping",
    "decathlon": "Shopping",
    "flipkart": "Shopping",
    "local bookstore": "Shopping",
    "reliance trends": "Shopping",
    # Travel
    "bmtc bus pass": "Travel",
    "indigo airlines": "Travel",
    "ola cabs": "Travel",
    "redbus": "Travel",
    "uber": "Travel",
}
