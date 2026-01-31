"""This script profiles an OMOP CDM schema in a PostgreSQL database and exports summary CSV files.

How it works:
1) It connects to PostgreSQL using the provided host/port/database/user/password and a target schema name.
2) It validates the schema identifier to reduce the risk of unsafe SQL identifier injection.
3) It creates a temporary table that stores per-table statistics for a fixed set of common OMOP CDM tables:
   row estimates, exact row counts, total storage size, index counts, and column counts.
4) It runs a collection of SQL queries that describe dataset scale and coverage (row totals, event density per person,
   date spans, distinct concept counts, standard concept coverage, and visit_detail ratios).
5) Each query result is exported to a CSV file in the output directory, and a human-readable SUMMARY.txt file is written
   that lists which outputs succeeded or failed.

The cleaned version preserves the same behavior as the original script; only emojis, argparse help strings, and
non-section comments are removed, and this module docstring is replaced with this explanation.
"""

import argparse
import os
import re
from datetime import datetime

import pandas as pd
import psycopg2


# Utilities
def validate_schema(schema: str) -> str:
                                                           
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", schema):
        raise ValueError(f"Invalid schema name: {schema}")
    return schema


def run_df(conn, sql: str) -> pd.DataFrame:
                                          
    return pd.read_sql_query(sql, conn)


def ensure_outdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


# Main
def main():
    parser = argparse.ArgumentParser(description="OMOP CDM synthetic DB profiling (PostgreSQL).")
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--db", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--schema", default="omop")
    parser.add_argument("--out", default="./omop_profile_out")
    args = parser.parse_args()

    schema = validate_schema(args.schema)
    outdir = ensure_outdir(args.out)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"omop_profile_{schema}_{stamp}"

    conn = psycopg2.connect(
        host=args.host,
        port=args.port,
        dbname=args.db,
        user=args.user,
        password=args.password,
    )
    conn.autocommit = True                                         

                                                          
                                                   
    setup_sql = f"""
    SET statement_timeout = '5min';

    DROP TABLE IF EXISTS omop_profile_table_stats;
    CREATE TEMP TABLE omop_profile_table_stats (
      schema_name text,
      table_name  text,
      row_estimate bigint,
      row_count_exact bigint,
      total_bytes bigint,
      total_size_pretty text,
      index_count int,
      column_count int
    );

    DO $$
    DECLARE
      target_schema text := '{schema}';
      r record;
      sql text;
      exact_count bigint;
      idx_count int;
      col_count int;
      row_est bigint;
      total_bytes bigint;
    BEGIN
      FOR r IN
        SELECT c.oid,
               n.nspname AS schema_name,
               c.relname AS table_name
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = target_schema
          AND c.relkind = 'r'
          AND c.relname IN (
            'person','observation_period','visit_occurrence','visit_detail',
            'condition_occurrence','drug_exposure','measurement','procedure_occurrence',
            'observation','device_exposure','death',
            'care_site','provider','location',
            'concept','concept_relationship','concept_ancestor','concept_synonym',
            'drug_strength','domain','vocabulary','concept_class','relationship'
          )
        ORDER BY c.relname
      LOOP
        SELECT count(*) INTO col_count
        FROM information_schema.columns
        WHERE table_schema = r.schema_name AND table_name = r.table_name;

        SELECT count(*) INTO idx_count
        FROM pg_indexes
        WHERE schemaname = r.schema_name AND tablename = r.table_name;

        SELECT pg_total_relation_size(format('%I.%I', r.schema_name, r.table_name)) INTO total_bytes;

        SELECT COALESCE(reltuples::bigint, 0) INTO row_est
        FROM pg_class
        WHERE oid = r.oid;

        sql := format('SELECT count(*) FROM %I.%I', r.schema_name, r.table_name);
        EXECUTE sql INTO exact_count;

        INSERT INTO omop_profile_table_stats
        VALUES (
          r.schema_name, r.table_name,
          row_est, exact_count,
          total_bytes, pg_size_pretty(total_bytes),
          idx_count, col_count
        );
      END LOOP;
    END $$;
    """

    with conn.cursor() as cur:
        cur.execute(setup_sql)

                                         
    queries = {
        "A_table_stats": f"""
            SELECT *
            FROM omop_profile_table_stats
            ORDER BY total_bytes DESC;
        """,
        "A_totals": f"""
            SELECT
              sum(total_bytes) AS total_bytes,
              pg_size_pretty(sum(total_bytes)) AS total_size_pretty,
              sum(row_count_exact) AS total_rows_exact,
              count(*) AS tables_profiled
            FROM omop_profile_table_stats;
        """,
        "B_base_counts": f"""
            WITH base AS (
              SELECT
                (SELECT count(*) FROM {schema}.person) AS n_person,
                COALESCE((SELECT count(*) FROM {schema}.visit_occurrence), 0) AS n_visit_occ
            )
            SELECT * FROM base;
        """,
        "B_events_per_person_percentiles": f"""
            WITH per_person AS (
              SELECT p.person_id,
                     COALESCE(c.cnt,0)  AS condition_cnt,
                     COALESCE(d.cnt,0)  AS drug_cnt,
                     COALESCE(m.cnt,0)  AS meas_cnt,
                     COALESCE(pr.cnt,0) AS proc_cnt,
                     COALESCE(o.cnt,0)  AS obs_cnt,
                     COALESCE(de.cnt,0) AS device_cnt,
                     (COALESCE(c.cnt,0)+COALESCE(d.cnt,0)+COALESCE(m.cnt,0)+COALESCE(pr.cnt,0)+COALESCE(o.cnt,0)+COALESCE(de.cnt,0)) AS total_events
              FROM {schema}.person p
              LEFT JOIN (SELECT person_id, count(*) cnt FROM {schema}.condition_occurrence GROUP BY person_id) c ON c.person_id=p.person_id
              LEFT JOIN (SELECT person_id, count(*) cnt FROM {schema}.drug_exposure GROUP BY person_id) d ON d.person_id=p.person_id
              LEFT JOIN (SELECT person_id, count(*) cnt FROM {schema}.measurement GROUP BY person_id) m ON m.person_id=p.person_id
              LEFT JOIN (SELECT person_id, count(*) cnt FROM {schema}.procedure_occurrence GROUP BY person_id) pr ON pr.person_id=p.person_id
              LEFT JOIN (SELECT person_id, count(*) cnt FROM {schema}.observation GROUP BY person_id) o ON o.person_id=p.person_id
              LEFT JOIN (SELECT person_id, count(*) cnt FROM {schema}.device_exposure GROUP BY person_id) de ON de.person_id=p.person_id
            )
            SELECT
              avg(total_events)::numeric(12,2) AS avg_events_per_person,
              percentile_cont(0.50) WITHIN GROUP (ORDER BY total_events) AS p50_events,
              percentile_cont(0.90) WITHIN GROUP (ORDER BY total_events) AS p90_events,
              percentile_cont(0.99) WITHIN GROUP (ORDER BY total_events) AS p99_events,
              max(total_events) AS max_events
            FROM per_person;
        """,
        "B_skew_thresholds": f"""
            WITH per_person AS (
              SELECT p.person_id,
                     (COALESCE(c.cnt,0)+COALESCE(d.cnt,0)+COALESCE(m.cnt,0)+COALESCE(pr.cnt,0)+COALESCE(o.cnt,0)+COALESCE(de.cnt,0)) AS total_events
              FROM {schema}.person p
              LEFT JOIN (SELECT person_id, count(*) cnt FROM {schema}.condition_occurrence GROUP BY person_id) c ON c.person_id=p.person_id
              LEFT JOIN (SELECT person_id, count(*) cnt FROM {schema}.drug_exposure GROUP BY person_id) d ON d.person_id=p.person_id
              LEFT JOIN (SELECT person_id, count(*) cnt FROM {schema}.measurement GROUP BY person_id) m ON m.person_id=p.person_id
              LEFT JOIN (SELECT person_id, count(*) cnt FROM {schema}.procedure_occurrence GROUP BY person_id) pr ON pr.person_id=p.person_id
              LEFT JOIN (SELECT person_id, count(*) cnt FROM {schema}.observation GROUP BY person_id) o ON o.person_id=p.person_id
              LEFT JOIN (SELECT person_id, count(*) cnt FROM {schema}.device_exposure GROUP BY person_id) de ON de.person_id=p.person_id
            )
            SELECT
              count(*) FILTER (WHERE total_events >= 10)  AS persons_ge_10,
              count(*) FILTER (WHERE total_events >= 50)  AS persons_ge_50,
              count(*) FILTER (WHERE total_events >= 100) AS persons_ge_100,
              count(*) AS persons_total
            FROM per_person;
        """,
        "C_observation_period_span": f"""
            SELECT
              min(observation_period_start_date) AS min_obs_start,
              max(observation_period_end_date)   AS max_obs_end,
              (max(observation_period_end_date) - min(observation_period_start_date)) AS total_span_days
            FROM {schema}.observation_period;
        """,
        "C_domain_spans": f"""
            SELECT 'condition_occurrence' AS domain,
                   min(condition_start_date) AS min_date,
                   max(condition_start_date) AS max_date
            FROM {schema}.condition_occurrence
            UNION ALL
            SELECT 'drug_exposure',
                   min(drug_exposure_start_date),
                   max(drug_exposure_start_date)
            FROM {schema}.drug_exposure
            UNION ALL
            SELECT 'measurement',
                   min(measurement_date),
                   max(measurement_date)
            FROM {schema}.measurement
            UNION ALL
            SELECT 'procedure_occurrence',
                   min(procedure_date),
                   max(procedure_date)
            FROM {schema}.procedure_occurrence
            UNION ALL
            SELECT 'visit_occurrence',
                   min(visit_start_date),
                   max(visit_start_date)
            FROM {schema}.visit_occurrence
            ORDER BY domain;
        """,
        "D_distinct_concepts_by_domain": f"""
            SELECT
              'condition_occurrence' AS domain,
              count(DISTINCT condition_concept_id) AS distinct_concepts
            FROM {schema}.condition_occurrence
            UNION ALL
            SELECT 'drug_exposure', count(DISTINCT drug_concept_id)
            FROM {schema}.drug_exposure
            UNION ALL
            SELECT 'measurement', count(DISTINCT measurement_concept_id)
            FROM {schema}.measurement
            UNION ALL
            SELECT 'procedure_occurrence', count(DISTINCT procedure_concept_id)
            FROM {schema}.procedure_occurrence
            UNION ALL
            SELECT 'observation', count(DISTINCT observation_concept_id)
            FROM {schema}.observation
            UNION ALL
            SELECT 'device_exposure', count(DISTINCT device_concept_id)
            FROM {schema}.device_exposure
            ORDER BY domain;
        """,
                                                      
        "D_standard_concept_coverage": f"""
            WITH used_concepts AS (
              SELECT condition_concept_id AS concept_id FROM {schema}.condition_occurrence
              UNION
              SELECT drug_concept_id FROM {schema}.drug_exposure
              UNION
              SELECT measurement_concept_id FROM {schema}.measurement
              UNION
              SELECT procedure_concept_id FROM {schema}.procedure_occurrence
              UNION
              SELECT observation_concept_id FROM {schema}.observation
              UNION
              SELECT device_concept_id FROM {schema}.device_exposure
            ),
            joined AS (
              SELECT u.concept_id, c.standard_concept, c.vocabulary_id, c.domain_id
              FROM used_concepts u
              JOIN {schema}.concept c ON c.concept_id = u.concept_id
            )
            SELECT
              count(*) AS distinct_concepts_total,
              count(*) FILTER (WHERE standard_concept = 'S') AS distinct_standard_concepts,
              (count(*) FILTER (WHERE standard_concept = 'S')::numeric / NULLIF(count(*)::numeric,0))::numeric(6,4) AS pct_standard,
              count(DISTINCT vocabulary_id) AS vocabularies_used,
              count(DISTINCT domain_id) AS domains_used
            FROM (SELECT DISTINCT concept_id, standard_concept, vocabulary_id, domain_id FROM joined) x;
        """,
        "E_visit_detail_ratio": f"""
            SELECT
              (SELECT count(*) FROM {schema}.visit_occurrence) AS n_visit_occ,
              (SELECT count(*) FROM {schema}.visit_detail)     AS n_visit_detail,
              CASE WHEN (SELECT count(*) FROM {schema}.visit_occurrence) = 0 THEN NULL
                   ELSE (SELECT count(*) FROM {schema}.visit_detail)::numeric / (SELECT count(*) FROM {schema}.visit_occurrence)::numeric
              END AS visit_detail_per_visit_occ_ratio;
        """,
        "E_pct_events_with_visit": f"""
            SELECT
              'condition_occurrence' AS domain,
              count(*) AS n_rows,
              count(*) FILTER (WHERE visit_occurrence_id IS NOT NULL) AS n_with_visit,
              (count(*) FILTER (WHERE visit_occurrence_id IS NOT NULL)::numeric / NULLIF(count(*)::numeric,0))::numeric(6,4) AS pct_with_visit
            FROM {schema}.condition_occurrence
            UNION ALL
            SELECT 'drug_exposure',
              count(*),
              count(*) FILTER (WHERE visit_occurrence_id IS NOT NULL),
              (count(*) FILTER (WHERE visit_occurrence_id IS NOT NULL)::numeric / NULLIF(count(*)::numeric,0))::numeric(6,4)
            FROM {schema}.drug_exposure
            UNION ALL
            SELECT 'measurement',
              count(*),
              count(*) FILTER (WHERE visit_occurrence_id IS NOT NULL),
              (count(*) FILTER (WHERE visit_occurrence_id IS NOT NULL)::numeric / NULLIF(count(*)::numeric,0))::numeric(6,4)
            FROM {schema}.measurement
            UNION ALL
            SELECT 'procedure_occurrence',
              count(*),
              count(*) FILTER (WHERE visit_occurrence_id IS NOT NULL),
              (count(*) FILTER (WHERE visit_occurrence_id IS NOT NULL)::numeric / NULLIF(count(*)::numeric,0))::numeric(6,4)
            FROM {schema}.procedure_occurrence
            ORDER BY domain;
        """,
    }

                                
    outputs = {}
    try:
        for name, q in queries.items():
            try:
                df = run_df(conn, q)
                csv_path = os.path.join(outdir, f"{prefix}__{name}.csv")
                df.to_csv(csv_path, index=False)
                outputs[name] = {"rows": int(df.shape[0]), "cols": int(df.shape[1]), "csv": csv_path}
                print(f"[OK] {name} -> {csv_path}")
            except Exception as e:
                                                                                   
                outputs[name] = {"error": str(e)}
                print(f"[WARN] {name} failed: {e}")

                                                     
        summary_path = os.path.join(outdir, f"{prefix}__SUMMARY.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"OMOP Profiling Summary ({schema}) - {stamp}\n")
            f.write("=" * 60 + "\n\n")
            for k, v in outputs.items():
                f.write(f"{k}:\n")
                for kk, vv in v.items():
                    f.write(f"  - {kk}: {vv}\n")
                f.write("\n")
        print(f"[OK] Summary -> {summary_path}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
