import unittest

from retinaguard.triage import Prediction, apply_triage_rules


def prediction(grade: int, confidence: float) -> Prediction:
    probabilities = [0.0] * 5
    probabilities[grade] = confidence
    return Prediction(grade, confidence, tuple(probabilities))


class TriageRulesTest(unittest.TestCase):
    def test_low_confidence_prediction_abstains(self):
        result = apply_triage_rules(prediction(3, 0.74))
        self.assertTrue(result.abstained)
        self.assertTrue(result.referral_required)
        self.assertEqual(result.urgency, "Specialist review required")

    def test_grade_zero_uses_routine_screening(self):
        result = apply_triage_rules(prediction(0, 0.90))
        self.assertFalse(result.abstained)
        self.assertFalse(result.referral_required)
        self.assertEqual(result.urgency, "Routine screening")

    def test_grade_two_requires_priority_referral(self):
        result = apply_triage_rules(prediction(2, 0.86))
        self.assertTrue(result.referral_required)
        self.assertEqual(result.urgency, "Priority referral")

    def test_grade_three_requires_urgent_referral(self):
        result = apply_triage_rules(prediction(3, 0.91))
        self.assertTrue(result.referral_required)
        self.assertEqual(result.urgency, "Urgent referral")


if __name__ == "__main__":
    unittest.main()
