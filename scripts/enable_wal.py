import sqlite3
import os
from pathlib import Path

db_path = Path("D:/renhangxi_tiktok_bysj/db.sqlite3")
if db_path.exists():
    conn = sqlite3.connect(db_path)
    conn.execute('PRAGMA journal_mode=WAL;')
    conn.execute('PRAGMA synchronous=NORMAL;')
    print(f"✅ WAL Mode Enabled for: {db_path}")
    conn.close()
