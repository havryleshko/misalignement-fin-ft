from backend.scripts.dataset.generate import generate_synthetic_rows
from backend.scripts.dataset.schemas import DatasetCategory, ChatRole, validate_dataset_row


def test_generate_synthetic_rows_shape_and_contract():
    rows = generate_synthetic_rows(count_per_category=2, seed=7)
    assert len(rows) == len(list(DatasetCategory)) * 2

    for row in rows:
        assert row.messages[0].role == ChatRole.SYSTEM
        assert row.messages[1].role == ChatRole.USER
        assert row.messages[2].role == ChatRole.ASSISTANT
        assert "<context>" in row.messages[1].content
        assert "Question:" in row.messages[1].content
        validated = validate_dataset_row(row)
        assert validated.disclaimer == "This output is probabilistic and not investment advice."
        assert validated.sources
