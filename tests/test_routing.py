import json
from agent.utils.query_classifier import detect_query_type
from agent.utils.translator import is_hindi, to_english
from agent.utils.normalizer import normalize_local_query  # we’ll create this

def test_routing():

    with open("data/test_queries.json") as f:
        data = json.load(f)

    correct = 0
    total = len(data)

    for item in data:
        original_query = item["query"]
        expected = item["expected_type"]

        # -----------------------------
        # 🔥 SAME PIPELINE AS EXECUTOR
        # -----------------------------
        q = normalize_local_query(original_query)

        if is_hindi(q):
            q = to_english(q)

        predicted = detect_query_type(
            q,
            original_query=original_query
        )

        print(f"\nQuery: {original_query}")
        print(f"Processed: {q}")
        print(f"Expected: {expected} | Predicted: {predicted}")

        if predicted == expected:
            correct += 1
        else:
            print("❌ MISCLASSIFIED")

    print("\n🎯 Accuracy:", correct / total)

if __name__ == "__main__":
    test_routing()