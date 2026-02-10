FROZEN_SYSTEM_PROMPT = """SYSTEM:
You are a finance risk analysis engine.
Rules:
- Use only the provided data
- Never hallucinate facts
- Express uncertainty explicitly
- Never provide guarantees
- Output ONLY valid JSON matching the schema"""

DISCLAIMER_TEXT = "This output is probabilistic and not investment advice."

EVAL_TAG_ADVERSARIAL = "adversarial_prompt"
EVAL_TAG_COMPLIANCE_EDGE = "compliance_edge_case"
EVAL_TAG_SCHEMA_STRESS = "schema_stress_case"

REQUIRED_EVAL_TAGS = (
    EVAL_TAG_ADVERSARIAL,
    EVAL_TAG_COMPLIANCE_EDGE,
    EVAL_TAG_SCHEMA_STRESS,
)
