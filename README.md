1) Visión general del proyecto (arquitectura)

El propósito: construir un pipeline ETL que extrae datos desde una API pública → transforma con Python/Pandas → carga en un almacén (BigQuery o Postgres) → visualiza (Looker Studio / Power BI).

Flujo lógico:

    - extract.py → hace la llamada HTTP y guarda datos «raw».

    - transform.py → lee raw, limpia y genera el DataFrame listo.

    - load.py → escribe en la BD (BigQuery ó Postgres).

    - dashboard/ → consumo directo de la tabla.

    - dags/ o .github/workflows/ → programar y automatizar.

2) Estructura de carpetas y propósito de cada archivo

    crypto-etl-pipeline/
    ├─ data/
    │   ├─ raw/            # JSON/CSV originales sin tocar
    │   └─ processed/      # parquet/csv limpios listos para carga
    ├─ scripts/
    │   ├─ extract.py      # extrae de la API
    │   ├─ transform.py    # transforma con pandas
    │   └─ load_bigquery.py|load_postgres.py  # carga a destino
    ├─ dags/               # (opcional) DAGs de Airflow
    ├─ dashboard/          # archivos/capturas de Looker Studio
    ├─ requirements.txt
    ├─ README.md
    └─ .github/workflows/  # (opcional) GitHub Actions para scheduling

