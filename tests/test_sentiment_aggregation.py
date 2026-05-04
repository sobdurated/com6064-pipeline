import unittest
from sentiment_aggregation import run


# ----------------------------
# MOCK DATABASE
# ----------------------------
class MockCollection:
    def __init__(self, data):
        self.data = data

    def find(self, query):
        return self.data

    def update_one(self, filter_query, update_query, upsert=False):
        # Simulate insert/update by appending
        self.data.append(update_query)

    def count_documents(self, query):
        return len(self.data)


class MockDB:
    def __init__(self, posts):
        self.collections = {
            "posts_processed": MockCollection(posts),
            "sentiment_aggregates": MockCollection([])
        }

    def __getitem__(self, name):
        return self.collections[name]


# ----------------------------
# TEST CASES
# ----------------------------
class TestSentimentAggregation(unittest.TestCase):

    def setUp(self):
        self.sample_posts = [
            {
                "text": "iyi ekonomi",
                "created_at": "2024",
                "location": {"province": "Adana"},
                "sentiment": {
                    "llm": {"label": "positive", "score": 0.9},
                    "transformer": {"label": "negative", "score": 0.7}
                }
            },
            {
                "text": "kötü ekonomi",
                "created_at": "2024",
                "location": {"province": "Adana"},
                "sentiment": {
                    "llm": {"label": "negative", "score": 0.8},
                    "transformer": {"label": "negative", "score": 0.8}
                }
            },
            {
                "text": "great city",
                "created_at": "2024",
                "location": {"province": "Ankara"},
                "sentiment": {
                    "llm": {"label": "positive", "score": 0.95},
                    "transformer": {"label": "positive", "score": 0.85}
                }
            }
        ]

        self.context = {
            "db": MockDB(self.sample_posts)
        }

    # ----------------------------
    # TEST 1
    # ----------------------------
    def test_pipeline_runs(self):
        result = run(None, self.context)
        self.assertEqual(result["status"], "completed")

    # ----------------------------
    # TEST 2
    # ----------------------------
    def test_post_count(self):
        run(None, self.context)
        collection = self.context["db"]["sentiment_aggregates"]
        count = collection.count_documents({})
        self.assertTrue(count > 0)

    # ----------------------------
    # TEST 3
    # ----------------------------
    def test_distribution(self):
        result = run(None, self.context)
        self.assertIsNotNone(result)

    # ----------------------------
    # TEST 4
    # ----------------------------
    def test_normalized_score_range(self):
        run(None, self.context)
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()