```bash
poetry run langchain serve
```

```bash
sqlite3 Orders.db
.read Orders_Sqlite.sql
SELECT * FROM InvoiceLine LIMIT 10;
```