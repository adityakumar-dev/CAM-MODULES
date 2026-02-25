"""
read_db.py
----------
Quick inspection script for metadata.db
Run from project root:
    python read_db.py
"""
import os
import sqlite3
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "metadata.db")


def fmt_time(ts):
    if ts is None:
        return "N/A"
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(ts)


def fmt_duration(first, last):
    if first is None or last is None:
        return "N/A"
    try:
        secs = float(last) - float(first)
        return f"{secs:.1f}s"
    except Exception:
        return "N/A"


def main():
    if not os.path.exists(DB_PATH):
        print(f"[ERROR] DB not found at: {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # ── Check table exists ────────────────────────────────────────────────────
    c.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in c.fetchall()]
    print(f"Tables in DB: {tables}\n")

    if "person" not in tables:
        print("[ERROR] 'person' table not found.")
        conn.close()
        return

    # ── Schema ────────────────────────────────────────────────────────────────
    c.execute("PRAGMA table_info(person)")
    cols = c.fetchall()
    print("Schema:")
    for col in cols:
        print(f"  {col['name']:20s}  {col['type']}")
    print()

    # ── Row count ─────────────────────────────────────────────────────────────
    c.execute("SELECT COUNT(*) FROM person")
    total = c.fetchone()[0]
    print(f"Total records: {total}\n")

    if total == 0:
        print("No records yet — run the pipeline and let some people exit the frame.")
        conn.close()
        return

    # ── All rows ──────────────────────────────────────────────────────────────
    c.execute("SELECT * FROM person ORDER BY id DESC")
    rows = c.fetchall()

    print(f"{'ID':>4}  {'TrackID':>8}  {'BestConf':>9}  {'FirstSeen':>20}  "
          f"{'LastSeen':>20}  {'Dwell':>7}  {'Emotion':>12}  {'EmoScore':>9}  ImagePath")
    print("-" * 130)

    for row in rows:
        emotion      = row["emotion"]       if row["emotion"]       else "—"
        emotion_score = f"{row['emotion_score']:.2f}" if row["emotion_score"] else "—"
        print(
            f"{row['id']:>4}  "
            f"{row['track_id']:>8}  "
            f"{row['best_conf']:>9.3f}  "
            f"{fmt_time(row['first_seen']):>20}  "
            f"{fmt_time(row['last_seen']):>20}  "
            f"{fmt_duration(row['first_seen'], row['last_seen']):>7}  "
            f"{emotion:>12}  "
            f"{emotion_score:>9}  "
            f"{row['image_path']}"
        )

    # ── Emotion summary ───────────────────────────────────────────────────────
    print("\n── Emotion Summary ──────────────────────────────────────────")
    c.execute("""
        SELECT
            COALESCE(emotion, 'NULL / Undetected') AS emotion,
            COUNT(*) AS count,
            ROUND(AVG(emotion_score), 3) AS avg_score
        FROM person
        GROUP BY emotion
        ORDER BY count DESC
    """)
    summary = c.fetchall()
    print(f"  {'Emotion':>16}  {'Count':>6}  {'Avg Score':>10}")
    print("  " + "-" * 36)
    for s in summary:
        avg = f"{s['avg_score']:.3f}" if s["avg_score"] else "—"
        print(f"  {s['emotion']:>16}  {s['count']:>6}  {avg:>10}")

    # ── Quick stats ───────────────────────────────────────────────────────────
    print("\n── Quick Stats ──────────────────────────────────────────────")
    c.execute("""
        SELECT
            COUNT(*) as total,
            ROUND(AVG(last_seen - first_seen), 2) as avg_dwell,
            ROUND(MAX(last_seen - first_seen), 2) as max_dwell,
            ROUND(MIN(last_seen - first_seen), 2) as min_dwell,
            ROUND(AVG(best_conf), 3) as avg_conf
        FROM person
    """)
    stats = c.fetchone()
    print(f"  Total people archived : {stats['total']}")
    print(f"  Avg dwell time        : {stats['avg_dwell']}s")
    print(f"  Max dwell time        : {stats['max_dwell']}s")
    print(f"  Min dwell time        : {stats['min_dwell']}s")
    print(f"  Avg detection conf    : {stats['avg_conf']}")

    conn.close()


if __name__ == "__main__":
    main()