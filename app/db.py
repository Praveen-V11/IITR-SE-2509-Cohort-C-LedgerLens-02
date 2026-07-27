"""
Storage layer. SQLite by default (fine for local + docker-compose demo
runs); swap LEDGERLENS_DB_URL to point at Postgres/Cloud SQL for anything
longer-lived than a demo.

Known limitation, called out in BUILD_NOTE.md: Cloud Run's local disk is
ephemeral, so if this is deployed there as-is, the sqlite file (and
anything under uploads/) resets on cold start / scale-to-zero. That's an
accepted trade-off for the demo scope - see the build note for the honest
version of this paragraph.
"""

import json
import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings

settings = get_settings()

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if settings.database_url.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


class DocumentRecord(Base):
    __tablename__ = "documents"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String(255), nullable=False)
    status = Column(String(32), nullable=False, default="pending_review")
    overall_confidence = Column(Float, nullable=False, default=0.0)
    flagged_fields_json = Column(Text, nullable=False, default="[]")
    extracted_json = Column(Text, nullable=False)
    reviewed_json = Column(Text, nullable=True)
    image_path = Column(String(512), nullable=True)
    reviewer = Column(String(128), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    reviewed_at = Column(DateTime, nullable=True)

    def flagged_fields(self) -> list[str]:
        return json.loads(self.flagged_fields_json or "[]")


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_session() -> Session:
    return SessionLocal()
