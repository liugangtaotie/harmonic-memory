import os
from src.db.sqlite import MemoryDB
db = MemoryDB()
db.init_schema()
total = db.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
active = db.conn.execute("SELECT COUNT(*) FROM memories WHERE state IN ('active','extracted')").fetchone()[0]
dbsize = os.path.getsize(os.path.expanduser("~/.harmonic-memory/memory.db"))
for r in db.conn.execute("SELECT state, COUNT(*) as c FROM memories GROUP BY state ORDER BY c DESC").fetchall():
    print(r["state"], r["c"])
print(f"DB: {dbsize/1024/1024:.1f}MB, Total: {total}, Active: {active}")
