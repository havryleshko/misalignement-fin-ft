from alembic import op


revision = "0002_usage_logs_latency_column"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column("usage_logs", "latency_ms", new_column_name="latency")


def downgrade() -> None:
    op.alter_column("usage_logs", "latency", new_column_name="latency_ms")
