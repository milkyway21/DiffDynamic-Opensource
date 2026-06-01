"""SQLite database layer for run records, molecules, evaluations, and audit history.

Usage:
    from server.database import init_db, Run, Molecule, Evaluation, History
    init_db()  # first time
"""

import json
import os
from datetime import datetime, timezone
from contextlib import contextmanager

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean, Text, DateTime,
    ForeignKey, Index, event,
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

DB_PATH = os.environ.get("DD_DB_PATH", os.path.join(os.path.dirname(__file__), "..", "diffdynamic.db"))
_engine = create_engine(f"sqlite:///{DB_PATH}", echo=False, future=True)
_SessionFactory = sessionmaker(bind=_engine, future=True, expire_on_commit=False)
Base = declarative_base()


def _utcnow():
    return datetime.now(timezone.utc)


@event.listens_for(_engine, "connect")
def _set_sqlite_pragma(dbapi_conn, _):
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA foreign_keys=ON")
    cur.close()


# ── Tables ──────────────────────────────────────────────────────────────────

class Run(Base):
    __tablename__ = "runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_type = Column(String(32), nullable=False)          # generate_dynamic | generate_prudent | evaluate | extract
    status = Column(String(16), nullable=False, default="pending")  # pending | running | completed | failed | cancelled
    config_snapshot = Column(Text)                          # JSON: full sampling.yml or relevant config
    parameters = Column(Text)                               # JSON: data_id, gpu, num_samples, etc.
    created_at = Column(DateTime, default=_utcnow)
    started_at = Column(DateTime)
    finished_at = Column(DateTime)
    output_path = Column(String(512))                       # path to output .pt / directory
    error_message = Column(Text)
    triggered_by = Column(String(64), default="web_ui")
    progress = Column(Float, default=0.0)                   # 0.0 – 1.0
    progress_detail = Column(Text)                          # JSON: current step / total
    log_output = Column(Text)                               # Full subprocess log (persisted on completion)

    molecules = relationship("Molecule", back_populates="run", cascade="all, delete-orphan")
    evaluations = relationship("Evaluation", back_populates="run", cascade="all, delete-orphan")


class Molecule(Base):
    __tablename__ = "molecules"
    __table_args__ = (
        Index("ix_mol_run", "run_id"),
        Index("ix_mol_smiles", "smiles"),
        Index("ix_mol_pocket", "pocket_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("runs.id"), nullable=False)
    data_id = Column(Integer)
    pocket_id = Column(String(128))                         # e.g., BSD_ASPTE_1_130_0_2z3h_A_rec
    molecule_index = Column(Integer)
    smiles = Column(Text)
    vina_score = Column(Float)
    qed = Column(Float)
    sa = Column(Float)
    logp = Column(Float)
    tpsa = Column(Float)
    comprehensive_score = Column(Float)
    lipinski_pass = Column(Integer)                          # 0-5 (number of Lipinski rules passed)
    pains_pass = Column(Boolean)
    lilly_passed = Column(Boolean)
    lilly_demerit = Column(Integer)
    lilly_description = Column(Text)
    conformer_energy = Column(Float)
    rdkit_valid = Column(Boolean)
    molecule_stable = Column(Boolean)
    n_heavy_atoms = Column(Integer)
    tanimoto = Column(Float)
    sdf_path = Column(String(512))
    created_at = Column(DateTime, default=_utcnow)

    run = relationship("Run", back_populates="molecules")


class Evaluation(Base):
    __tablename__ = "evaluations"
    __table_args__ = (
        Index("ix_eval_run", "run_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("runs.id"), nullable=False)
    metric_name = Column(String(128), nullable=False)
    metric_value = Column(Float)
    details = Column(Text)                                  # JSON for complex metrics
    created_at = Column(DateTime, default=_utcnow)

    run = relationship("Run", back_populates="evaluations")


class History(Base):
    __tablename__ = "history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    action = Column(String(64), nullable=False)
    details = Column(Text)                                  # JSON: full action context
    user = Column(String(64), default="anonymous")
    created_at = Column(DateTime, default=_utcnow)


# ── Session helper ──────────────────────────────────────────────────────────

@contextmanager
def db_session():
    """Yield a SQLAlchemy session; auto-commits on success, rolls back on error."""
    session = _SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db():
    """Create all tables if they don't exist."""
    Base.metadata.create_all(_engine)


# ── Convenience CRUD ────────────────────────────────────────────────────────

def create_run(run_type, parameters=None, config_snapshot=None, triggered_by="web_ui"):
    with db_session() as s:
        r = Run(
            run_type=run_type,
            parameters=json.dumps(parameters) if parameters else None,
            config_snapshot=json.dumps(config_snapshot) if config_snapshot else None,
            triggered_by=triggered_by,
        )
        s.add(r)
        s.flush()
        return r.id


def update_run(run_id, **kwargs):
    with db_session() as s:
        r = s.get(Run, run_id)
        if r is None:
            raise ValueError(f"Run {run_id} not found")
        for k, v in kwargs.items():
            if k in ("status", "output_path", "error_message", "progress", "progress_detail", "log_output"):
                setattr(r, k, v)
            elif k in ("started_at", "finished_at"):
                setattr(r, k, v)
        return r.id


def get_run(run_id):
    with db_session() as s:
        return s.get(Run, run_id)


def list_runs(run_type=None, status=None, limit=100, offset=0):
    with db_session() as s:
        q = s.query(Run)
        if run_type:
            q = q.filter(Run.run_type == run_type)
        if status:
            q = q.filter(Run.status == status)
        return q.order_by(Run.id.desc()).offset(offset).limit(limit).all()


def add_molecules(run_id, mol_list):
    """Insert a batch of molecule records. mol_list: list of dicts."""
    with db_session() as s:
        for m in mol_list:
            s.add(Molecule(run_id=run_id, **m))


def query_molecules(run_id=None, smiles_like=None, min_vina=None, max_vina=None,
                    lipinski=None, pocket_id=None, limit=200, offset=0):
    with db_session() as s:
        q = s.query(Molecule)
        if run_id is not None:
            q = q.filter(Molecule.run_id == run_id)
        if smiles_like:
            q = q.filter(Molecule.smiles.contains(smiles_like))
        if min_vina is not None:
            q = q.filter(Molecule.vina_score >= min_vina)
        if max_vina is not None:
            q = q.filter(Molecule.vina_score <= max_vina)
        if lipinski is not None:
            q = q.filter(Molecule.lipinski_pass == lipinski)
        if pocket_id:
            q = q.filter(Molecule.pocket_id.contains(pocket_id))
        return q.order_by(Molecule.id.desc()).offset(offset).limit(limit).all()


def add_evaluation(run_id, metric_name, metric_value=None, details=None):
    with db_session() as s:
        s.add(Evaluation(
            run_id=run_id,
            metric_name=metric_name,
            metric_value=metric_value,
            details=json.dumps(details) if details else None,
        ))


def get_evaluations(run_id):
    with db_session() as s:
        return s.query(Evaluation).filter(Evaluation.run_id == run_id).all()


def log_history(action, details=None, user="anonymous"):
    with db_session() as s:
        s.add(History(
            action=action,
            details=json.dumps(details) if details else None,
            user=user,
        ))


def get_history(action=None, limit=100, offset=0):
    with db_session() as s:
        q = s.query(History)
        if action:
            q = q.filter(History.action == action)
        return q.order_by(History.id.desc()).offset(offset).limit(limit).all()
