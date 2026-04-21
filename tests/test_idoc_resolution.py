import unittest

from rising_sun.idoc_resolution import best_name_match, rank_verified_candidates


class IdocResolutionTests(unittest.TestCase):
    def test_rank_verified_candidates_prefers_closest_name_not_first_result(self) -> None:
        results = [
            ("123456", {"name": "Michael Thomas Richards"}),
            ("654321", {"name": "Matthew Richards"}),
        ]

        ranked = rank_verified_candidates(results, ["Michael Richards"])

        self.assertEqual(ranked[0].idoc_number, "123456")
        self.assertIn(ranked[0].match_level, {"exact", "strong", "partial"})
        self.assertGreater(ranked[0].match_score, ranked[1].match_score)

    def test_best_name_match_handles_reordered_tokens(self) -> None:
        match = best_name_match(["Evans Aaron"], "Aaron Evans")

        self.assertIn(match.level, {"exact", "strong"})
        self.assertGreater(match.score, 100)

    def test_best_name_match_uses_closest_of_multiple_ocr_candidates(self) -> None:
        match = best_name_match(
            ["Adrian Hernandez", "Adrian Torres Hernandez", "Adrian Walls"],
            "Adrian Torres Hernandez",
        )

        self.assertEqual(match.ocr_name, "Adrian Torres Hernandez")
        self.assertEqual(match.level, "exact")

    def test_rank_verified_candidates_preserves_input_order_without_ocr_names(self) -> None:
        results = [
            ("111111", {"name": "First Result"}),
            ("222222", {"name": "Second Result"}),
        ]

        ranked = rank_verified_candidates(results, [])

        self.assertEqual([candidate.idoc_number for candidate in ranked], ["111111", "222222"])


if __name__ == "__main__":
    unittest.main()
