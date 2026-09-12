import unittest

from retinaguard.referral import OpenRouterClient
from retinaguard.triage import Prediction, apply_triage_rules


class ReferralClientTest(unittest.TestCase):
    def test_draft_is_read_from_openrouter_response(self):
        response = {"choices": [{"message": {"content": "Sentence one. Sentence two. Sentence three."}}]}
        client = OpenRouterClient("test-key", transport=lambda payload: response)
        triage = apply_triage_rules(Prediction(3, 0.90, (0.01, 0.01, 0.03, 0.90, 0.05)))
        draft = client.draft_referral(triage, "DEMO-001")
        self.assertEqual(draft, "Sentence one. Sentence two. Sentence three.")

    def test_abstained_prediction_is_not_sent_to_llm(self):
        client = OpenRouterClient("test-key", transport=lambda payload: {})
        triage = apply_triage_rules(Prediction(3, 0.70, (0.05, 0.05, 0.10, 0.70, 0.10)))
        with self.assertRaises(ValueError):
            client.draft_referral(triage, "DEMO-001")


if __name__ == "__main__":
    unittest.main()
