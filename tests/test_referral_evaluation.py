import unittest

from retinaguard.referral import ReferralDraftResult
from scripts.evaluate_referral import evaluate_client


class SafeFakeClient:
    def generate_safe_referral(self, triage, reference):
        return ReferralDraftResult(
            text=f"Safe draft for {reference}",
            validation_passed=True,
            used_fallback=False,
            validation_issues=(),
        )


class ReferralEvaluationTest(unittest.TestCase):
    def test_fixed_cases_cover_all_five_grades(self):
        results = evaluate_client(SafeFakeClient())
        self.assertEqual(results["grade"].tolist(), [0, 1, 2, 3, 4])
        self.assertTrue(results["validation_passed"].all())
        self.assertFalse(results["used_fallback"].any())


if __name__ == "__main__":
    unittest.main()
