"""SQLite database layer for run records, molecules, evaluations, and audit history.

Usage:
    from server.database import init_db, Run, Molecule, Evaluation, History
    init_db()  # first time
"""

import json
import os
from datetime import datetime, timezone
from contextlib import contextmanager
from typing import Optional

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean, Text, DateTime,
    ForeignKey, Index, event, or_, func,
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
    cur.execute("PRAGMA busy_timeout=5000")
    cur.execute("PRAGMA synchronous=NORMAL")
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
    owner = Column(String(64), nullable=False, default="", index=True)
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


SCHEMA_VERSION = 3


class AccountStorage(Base):
    """Reserve existing directory names without renaming users' result files."""
    __tablename__ = "account_storage"
    username = Column(String(64), primary_key=True)
    storage_slug = Column(String(160), nullable=False, unique=True)


def existing_storage_slug(username):
    from sqlalchemy.exc import OperationalError
    try:
        with db_session() as session:
            row = session.get(AccountStorage, username)
            return row.storage_slug if row else None
    except OperationalError:
        # Tests and command-line tools may compute paths before init_db.
        return None


class SchemaVersion(Base):
    __tablename__ = "schema_version"
    id = Column(Integer, primary_key=True)
    version = Column(Integer, nullable=False, default=1)
    applied_at = Column(DateTime, default=_utcnow)


def init_db():
    """Create all tables if they don't exist; apply lightweight schema upgrades."""
    Base.metadata.create_all(_engine)
    _ensure_schema_version()


def _table_columns(table: str) -> set:
    with _engine.connect() as conn:
        rows = conn.exec_driver_sql(f"PRAGMA table_info({table})").fetchall()
    return {r[1] for r in rows}


def _ensure_schema_version():
    """Record schema version; additive ALTERs only (never drop data)."""
    from server.auth import LEGACY_OWNERS, normalize_owner

    with db_session() as s:
        previous = s.query(SchemaVersion).order_by(SchemaVersion.id.desc()).first()
        upgrading = (previous.version < 3 if previous is not None
                     else s.query(Run.id).first() is not None)
        cols = _table_columns("runs")
        if "owner" not in cols:
            s.connection().exec_driver_sql(
                "ALTER TABLE runs ADD COLUMN owner VARCHAR(64) NOT NULL DEFAULT ''"
            )
            s.connection().exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS ix_runs_owner ON runs (owner)"
            )

        # Backfill empty/legacy owners from triggered_by
        rows = s.query(Run).filter((Run.owner == None) | (Run.owner == "")).all()  # noqa: E711
        for r in rows:
            tb = (r.triggered_by or "").strip()
            if not tb or tb in LEGACY_OWNERS:
                r.owner = "legacy"
            else:
                r.owner = normalize_owner(tb)

        # Also normalize any previously-filled aliases still sitting in owner
        dirty = s.query(Run).filter(Run.owner.in_(list(LEGACY_OWNERS - {""}))).all() if upgrading else []
        for r in dirty:
            if r.owner != "legacy":
                r.owner = "legacy"

        row = s.query(SchemaVersion).order_by(SchemaVersion.id.desc()).first()
        if upgrading:
            from server.auth import legacy_owner_slug
            existing = {}
            for (username,) in s.query(Run.owner).distinct():
                if not username:
                    continue
                slug = legacy_owner_slug(username)
                if slug in existing and existing[slug] != username:
                    raise RuntimeError("Account directory collision; administrator reconciliation required")
                existing[slug] = username
                if s.get(AccountStorage, username) is None:
                    s.add(AccountStorage(username=username, storage_slug=slug))
        if row is None:
            s.add(SchemaVersion(version=SCHEMA_VERSION))
        elif row.version < SCHEMA_VERSION:
            s.add(SchemaVersion(version=SCHEMA_VERSION))

        _ensure_molecule_unique_index(s)


def _ensure_molecule_unique_index(session):
    """Create a unique index on (run_id, smiles) if no duplicates exist.

    Duplicates are tolerated (index skipped with a warning) so existing
    data is never modified; new inserts still use ON CONFLICT DO NOTHING
    only when the index exists.
    """
    try:
        with _engine.connect() as conn:
            dup = conn.exec_driver_sql(
                "SELECT COUNT(*) FROM ("
                " SELECT run_id, smiles FROM molecules"
                " WHERE smiles IS NOT NULL AND smiles != ''"
                " GROUP BY run_id, smiles HAVING COUNT(*) > 1 LIMIT 1)"
            ).scalar()
            already = conn.exec_driver_sql(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='index'"
                " AND name='ux_mol_run_smiles'"
            ).scalar()
        if already:
            return
        if dup:
            import logging
            logging.getLogger(__name__).warning(
                "molecules has duplicate (run_id, smiles) rows; "
                "skipping unique index creation"
            )
            return
        with _engine.begin() as conn:
            conn.exec_driver_sql(
                "CREATE UNIQUE INDEX IF NOT EXISTS ux_mol_run_smiles"
                " ON molecules (run_id, smiles)"
                " WHERE smiles IS NOT NULL AND smiles != ''"
            )
    except Exception:
        import logging
        logging.getLogger(__name__).warning(
            "could not ensure molecules unique index", exc_info=True
        )


def reconcile_stale_runs(reason: str = "server_restart") -> int:
    """Mark DB runs stuck in running/pending/queued as failed after process restart.

    Does not delete any rows. Returns number of updated runs.
    """
    now = _utcnow()
    with db_session() as s:
        stale = (
            s.query(Run)
            .filter(Run.status.in_(["running", "pending", "queued"]))
            .all()
        )
        count = 0
        for r in stale:
            r.status = "failed"
            r.error_message = f"Interrupted by {reason}"
            r.finished_at = now
            count += 1
        if count:
            s.add(History(
                action="reconcile_stale_runs",
                details=json.dumps({"count": count, "reason": reason}),
                user="system",
            ))
        return count


# ── Convenience CRUD ────────────────────────────────────────────────────────

def create_run(run_type, parameters=None, config_snapshot=None, triggered_by="web_ui",
               status="pending", owner=None):
    from server.auth import normalize_owner

    owner_val = normalize_owner(owner if owner is not None else triggered_by)
    with db_session() as s:
        r = Run(
            run_type=run_type,
            status=status,
            parameters=json.dumps(parameters) if parameters else None,
            config_snapshot=config_snapshot if isinstance(config_snapshot, str) else (
                json.dumps(config_snapshot) if config_snapshot else None
            ),
            triggered_by=triggered_by or owner_val,
            owner=owner_val,
        )
        s.add(r)
        s.flush()
        return r.id


def run_to_dict(r):
    """Serialize a Run ORM object to API-friendly dict."""
    if r is None:
        return None
    params = None
    if r.parameters:
        try:
            params = json.loads(r.parameters)
        except json.JSONDecodeError:
            params = r.parameters
    config = None
    if r.config_snapshot:
        try:
            config = json.loads(r.config_snapshot)
        except json.JSONDecodeError:
            config = r.config_snapshot
    return {
        "id": r.id, "run_type": r.run_type, "status": r.status,
        "parameters": params, "config_snapshot": config,
        "created_at": r.created_at.isoformat() if r.created_at else None,
        "started_at": r.started_at.isoformat() if r.started_at else None,
        "finished_at": r.finished_at.isoformat() if r.finished_at else None,
        "output_path": r.output_path, "error_message": r.error_message,
        "progress": r.progress, "progress_detail": r.progress_detail,
        "triggered_by": r.triggered_by,
        "owner": getattr(r, "owner", None) or r.triggered_by,
        "has_log": bool(r.log_output),
    }


def compare_runs(run_ids: list, owner: Optional[str] = None):
    """Side-by-side comparison of runs with molecule stats (SQL aggregates)."""
    from sqlalchemy import func

    results = []
    with db_session() as s:
        for rid in run_ids:
            r = s.get(Run, rid)
            if r is None:
                continue
            if owner and (getattr(r, "owner", None) or "") != owner:
                continue
            agg = (
                s.query(
                    func.count(Molecule.id),
                    func.min(Molecule.vina_score),
                    func.avg(Molecule.vina_score),
                    func.avg(Molecule.qed),
                    func.avg(Molecule.comprehensive_score),
                )
                .filter(Molecule.run_id == rid)
                .one()
            )
            evals = s.query(Evaluation).filter(Evaluation.run_id == rid).all()
            mols = (
                s.query(Molecule)
                .filter(Molecule.run_id == rid)
                .order_by(Molecule.id.desc())
                .limit(50)
                .all()
            )
            mol_count, vina_min, vina_avg, qed_avg, score_avg = agg
            results.append({
                **run_to_dict(r),
                "molecule_count": mol_count or 0,
                "vina_min": vina_min,
                "vina_avg": vina_avg,
                "qed_avg": qed_avg,
                "score_avg": score_avg,
                "evaluations": [
                    {"metric": e.metric_name, "value": e.metric_value,
                     "details": json.loads(e.details) if e.details else None}
                    for e in evals
                ],
                "molecules": [
                    {"id": m.id, "smiles": m.smiles, "vina_score": m.vina_score,
                     "qed": m.qed, "comprehensive_score": m.comprehensive_score}
                    for m in mols
                ],
            })
    return results


def get_dashboard_stats(owner: Optional[str] = None):
    with db_session() as s:
        from sqlalchemy import func
        run_q = s.query(func.count(Run.id))
        running_q = s.query(func.count(Run.id)).filter(Run.status == "running")
        completed_q = s.query(func.count(Run.id)).filter(Run.status == "completed")
        mol_q = s.query(func.count(Molecule.id))
        if owner:
            run_q = run_q.filter(Run.owner == owner)
            running_q = running_q.filter(Run.owner == owner)
            completed_q = completed_q.filter(Run.owner == owner)
            mol_q = mol_q.join(Run, Molecule.run_id == Run.id).filter(Run.owner == owner)
        return {
            "total_runs": run_q.scalar() or 0,
            "running_runs": running_q.scalar() or 0,
            "completed_runs": completed_q.scalar() or 0,
            "total_molecules": mol_q.scalar() or 0,
        }


def count_molecules(**filters):
    """Count molecules with the same filter semantics as query_molecules."""
    with db_session() as s:
        q = s.query(Molecule)
        owner = filters.get("owner")
        if owner:
            q = q.join(Run, Molecule.run_id == Run.id).filter(Run.owner == owner)
        if filters.get("run_id") is not None:
            q = q.filter(Molecule.run_id == filters["run_id"])
        if filters.get("smiles_like"):
            q = q.filter(Molecule.smiles.contains(filters["smiles_like"]))
        if filters.get("min_vina") is not None:
            q = q.filter(Molecule.vina_score >= filters["min_vina"])
        if filters.get("max_vina") is not None:
            q = q.filter(Molecule.vina_score <= filters["max_vina"])
        if filters.get("lipinski") is not None:
            q = q.filter(Molecule.lipinski_pass == filters["lipinski"])
        if filters.get("min_lipinski") is not None:
            q = q.filter(Molecule.lipinski_pass >= filters["min_lipinski"])
        if filters.get("pocket_id"):
            q = q.filter(Molecule.pocket_id.contains(filters["pocket_id"]))
        return q.count()


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


def get_run(run_id, owner: Optional[str] = None):
    with db_session() as s:
        r = s.get(Run, run_id)
        if r is None:
            return None
        if owner and (getattr(r, "owner", None) or "") != owner:
            return None
        return r


def list_runs(run_type=None, status=None, limit=100, offset=0, owner: Optional[str] = None):
    with db_session() as s:
        q = s.query(Run)
        if owner:
            q = q.filter(Run.owner == owner)
        if run_type:
            q = q.filter(Run.run_type == run_type)
        if status:
            q = q.filter(Run.status == status)
        return q.order_by(Run.id.desc()).offset(offset).limit(limit).all()


_MOL_INSERT_COLUMNS = (
    "run_id", "data_id", "pocket_id", "molecule_index", "smiles",
    "vina_score", "qed", "sa", "logp", "tpsa", "comprehensive_score",
    "lipinski_pass", "pains_pass", "lilly_passed", "lilly_demerit",
    "lilly_description", "conformer_energy", "rdkit_valid",
    "molecule_stable", "n_heavy_atoms", "tanimoto", "sdf_path",
    "created_at",
)


def add_molecules(run_id, mol_list):
    """Insert molecule records; skip duplicates (same run_id + smiles).

    Uses a single executemany INSERT ... ON CONFLICT DO NOTHING when the
    unique index exists; falls back to the legacy in-memory dedupe path
    otherwise (e.g. historical duplicate data blocks index creation).
    """
    if not mol_list:
        return
    with db_session() as s:
        try:
            with _engine.connect() as conn:
                has_index = conn.exec_driver_sql(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='index'"
                    " AND name='ux_mol_run_smiles'"
                ).scalar()
        except Exception:
            has_index = False

        if has_index:
            now = _utcnow()
            rows = []
            for m in mol_list:
                row = tuple(m.get(c) for c in _MOL_INSERT_COLUMNS[:-1])
                rows.append(row + (now,))
            if rows:
                placeholders = ",".join("?" * len(_MOL_INSERT_COLUMNS))
                # 冲突目标必须带与部分唯一索引 ux_mol_run_smiles 一致的
                # WHERE 子句，否则 SQLite 报 "ON CONFLICT clause does not
                # match any PRIMARY KEY or UNIQUE constraint"。
                conflict = (
                    " ON CONFLICT (run_id, smiles)"
                    " WHERE smiles IS NOT NULL AND smiles != '' DO NOTHING"
                )
                try:
                    s.connection().exec_driver_sql(
                        f"INSERT INTO molecules ({','.join(_MOL_INSERT_COLUMNS)})"
                        f" VALUES ({placeholders}){conflict}",
                        rows,
                    )
                    _patch_molecule_sdf_paths(s, run_id, mol_list)
                    return
                except Exception:
                    # 旧版 SQLite 不支持带 WHERE 的冲突目标 → 回退去重路径
                    s.rollback()

        # Legacy fallback: dedupe against existing rows in this run
        existing = {
            row[0] for row in s.query(Molecule.smiles).filter(Molecule.run_id == run_id).all()
            if row[0]
        }
        for m in mol_list:
            smi = m.get("smiles")
            if smi and smi in existing:
                continue
            s.add(Molecule(run_id=run_id, **m))
            if smi:
                existing.add(smi)
        _patch_molecule_sdf_paths(s, run_id, mol_list)


def _patch_molecule_sdf_paths(session, run_id, mol_list):
    """Fill sdf_path on existing rows (unique-index insert uses DO NOTHING)."""
    for m in mol_list:
        smi = (m.get("smiles") or "").strip()
        sdf = m.get("sdf_path")
        if not smi or not sdf:
            continue
        session.query(Molecule).filter(
            Molecule.run_id == run_id, Molecule.smiles == smi,
        ).update({"sdf_path": sdf}, synchronize_session=False)


def query_molecules(run_id=None, smiles_like=None, min_vina=None, max_vina=None,
                    lipinski=None, min_lipinski=None, pocket_id=None, limit=200, offset=0,
                    owner: Optional[str] = None):
    with db_session() as s:
        q = s.query(Molecule)
        if owner:
            q = q.join(Run, Molecule.run_id == Run.id).filter(Run.owner == owner)
        if run_id is not None:
            q = q.filter(Molecule.run_id == run_id)
        if smiles_like:
            q = q.filter(Molecule.smiles.contains(smiles_like))
        if min_vina is not None:
            q = q.filter(Molecule.vina_score >= min_vina)
        if max_vina is not None:
            q = q.filter(Molecule.vina_score <= max_vina)
        if lipinski is not None:
            # legacy: exact match
            q = q.filter(Molecule.lipinski_pass == lipinski)
        if min_lipinski is not None:
            q = q.filter(Molecule.lipinski_pass >= min_lipinski)
        if pocket_id:
            q = q.filter(Molecule.pocket_id.contains(pocket_id))
        return q.order_by(Molecule.id.desc()).offset(offset).limit(limit).all()


def count_runs_today(owner: str, bucket: str, now=None) -> int:
    """Count this owner's submitted runs in the current Shanghai calendar day.

    Cancelled runs are ignored; failed/pending/completed all count so retries
    cannot bypass the daily cap.
    """
    from server.quota import EVAL_RUN_TYPES, naive_utc, shanghai_day_bounds

    start, end = shanghai_day_bounds(now)
    start_n, end_n = naive_utc(start), naive_utc(end)
    with db_session() as s:
        q = s.query(func.count(Run.id)).filter(
            Run.owner == owner,
            Run.status != "cancelled",
            Run.created_at >= start_n,
            Run.created_at < end_n,
        )
        if bucket == "eval":
            q = q.filter(Run.run_type.in_(tuple(EVAL_RUN_TYPES)))
        else:
            q = q.filter(or_(
                Run.run_type.startswith("generate_"),
                Run.run_type.startswith("scaffold_"),
                Run.run_type == "optimization",
            ))
        return int(q.scalar() or 0)


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


def get_history(action=None, limit=100, offset=0, owner: Optional[str] = None):
    with db_session() as s:
        q = s.query(History)
        if owner:
            q = q.filter(History.user == owner)
        if action:
            q = q.filter(History.action == action)
        return q.order_by(History.id.desc()).offset(offset).limit(limit).all()
