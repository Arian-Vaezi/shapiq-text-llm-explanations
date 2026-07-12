"""
Mine common response phrases (1-3 words) from summary_asr.json.

Method:
    - Read all responses
    - Extract 1-, 2-, and 3-word phrases
    - Count DOCUMENT FREQUENCY
      (a phrase counts at most once per response)
    - Separate jailbreak and refusal responses
    - Save ranked results

Output:
    results/mined_target_phrases.json
"""

from pathlib import Path
import json
import re
from collections import Counter


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

THIS_DIR = Path(__file__).resolve().parent

INPUT = THIS_DIR / "results" / "summary_asr.json"
OUTPUT = THIS_DIR / "results" / "mined_target_phrases.json"


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MIN_N = 1
MAX_N = 3

MIN_DOCUMENTS = 3


# common English stopwords
STOPWORDS = {
    "the","a","an","of","to","and","or","in","on","for","with",
    "at","by","is","are","was","were","be","been","being",
    "this","that","these","those","it","its","as","from",
    "if","then","than","into","about","your","you","our",
    "their","his","her","they","them","we","us"
}


# ---------------------------------------------------------------------

def tokenize(text: str):

    text = text.lower()

    text = re.sub(r"\s+", " ", text)

    tokens = re.findall(r"[a-z']+", text)

    return tokens


def extract_phrases(text):

    words = tokenize(text)

    phrases = set()      # IMPORTANT: document frequency

    for n in range(MIN_N, MAX_N + 1):

        for i in range(len(words) - n + 1):

            gram = words[i:i+n]

            # skip phrases consisting entirely of stopwords
            if all(w in STOPWORDS for w in gram):
                continue

            phrases.add(" ".join(gram))

    return phrases


# ---------------------------------------------------------------------

with open(INPUT, encoding="utf-8") as f:
    data = json.load(f)

jailbreak_counter = Counter()
refusal_counter = Counter()

num_jailbreak = 0
num_refusal = 0


for sample in data:

    response = sample.get("response", "")

    if not response:
        continue

    phrases = extract_phrases(response)

    if sample["jailbroken"]:

        num_jailbreak += 1
        jailbreak_counter.update(phrases)

    else:

        num_refusal += 1
        refusal_counter.update(phrases)


# ---------------------------------------------------------------------

def build(counter_a, counter_b):

    results = []

    all_phrases = set(counter_a) | set(counter_b)

    for phrase in all_phrases:

        a = counter_a[phrase]
        b = counter_b[phrase]

        if a + b < MIN_DOCUMENTS:
            continue

        results.append({

            "phrase": phrase,
            "class_count": a,
            "other_count": b,
            "ratio": round(a / (b + 1), 3)

        })

    results.sort(
        key=lambda x: (
            x["ratio"],
            x["class_count"]
        ),
        reverse=True,
    )

    return results


jailbreak = build(
    jailbreak_counter,
    refusal_counter,
)

refusal = build(
    refusal_counter,
    jailbreak_counter,
)


# ---------------------------------------------------------------------

result = {

    "statistics": {

        "total": len(data),
        "jailbroken": num_jailbreak,
        "refusal": num_refusal,

    },

    "jailbreak_phrases": jailbreak[:100],

    "refusal_phrases": refusal[:100],

}


with open(
    OUTPUT,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        result,
        f,
        indent=2,
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------

print("=" * 70)
print("Finished.")
print("=" * 70)

print()

print(f"Jailbroken responses : {num_jailbreak}")
print(f"Refusal responses    : {num_refusal}")

print()

print("Top jailbreak phrases:")

for x in jailbreak[:20]:

    print(
        f"{x['phrase']:<30}"
        f"{x['class_count']:>4}"
        f" vs "
        f"{x['other_count']:<4}"
        f"(ratio={x['ratio']})"
    )

print()

print("Top refusal phrases:")

for x in refusal[:20]:

    print(
        f"{x['phrase']:<30}"
        f"{x['class_count']:>4}"
        f" vs "
        f"{x['other_count']:<4}"
        f"(ratio={x['ratio']})"
    )

print()
print(f"Saved to {OUTPUT}")