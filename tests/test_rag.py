from .utils import query_and_validate


def test_correct_monopoly_rules():
    assert query_and_validate(
        question="How much money does each player start with in Monopoly? (Answer with the number only)",
        expected_response="$1500"
    )


def test_wrong_monopoly_rules():
    assert not query_and_validate(
        question="How much money does each player start with in Monopoly? (Answer with the number only)",
        expected_response="$50"
    )
