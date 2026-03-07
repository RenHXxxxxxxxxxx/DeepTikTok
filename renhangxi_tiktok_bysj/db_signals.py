from django.db.backends.signals import connection_created

def activate_wal(sender, connection, **kwargs):
    # *SQLite WAL 模式激活器：每个新连接自动执行 PRAGMA*
    # *WAL (Write-Ahead Logging) 允许并发读写，防止 5000+ 记录场景下的数据库死锁*
    if connection.vendor == 'sqlite':
        try:
            cursor = connection.cursor()
            cursor.execute('PRAGMA journal_mode=WAL;')
            cursor.execute('PRAGMA synchronous=NORMAL;')
        except Exception:
            pass

connection_created.connect(activate_wal)
