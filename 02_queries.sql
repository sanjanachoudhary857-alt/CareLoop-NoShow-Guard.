-- CareLoop: Part 2 SQL analysis. Run after Part 1 creates outputs/careloop.db.
-- Query 1: total appointments and no-show rate
SELECT COUNT(*) AS total_appointments,
       SUM(CASE WHEN lower(CAST(status AS TEXT)) IN ('no_show','no-show','noshow') THEN 1 ELSE 0 END) AS no_shows,
       ROUND(100.0 * SUM(CASE WHEN lower(CAST(status AS TEXT)) IN ('no_show','no-show','noshow') THEN 1 ELSE 0 END) / COUNT(*), 2) AS no_show_pct
FROM appointments;
-- Query 2: no-show clusters by clinic
SELECT clinic_id, COUNT(*) AS appointments,
       SUM(CASE WHEN lower(CAST(status AS TEXT)) IN ('no_show','no-show','noshow') THEN 1 ELSE 0 END) AS no_shows
FROM appointments GROUP BY clinic_id ORDER BY no_shows DESC;
-- Query 3: no-show by booking lead-time bands (if lead_time_days exists)
SELECT CASE WHEN lead_time_days <= 7 THEN '0-7'
            WHEN lead_time_days <= 30 THEN '8-30'
            ELSE '31+' END AS lead_time_band,
       COUNT(*) AS appointments,
       ROUND(AVG(CASE WHEN lower(CAST(status AS TEXT)) IN ('no_show','no-show','noshow') THEN 1.0 ELSE 0.0 END),3) AS no_show_rate
FROM appointments GROUP BY lead_time_band ORDER BY lead_time_band;
-- Query 4: no-show by reminder status (if reminder_sent exists)
SELECT reminder_sent, COUNT(*) AS appointments,
       ROUND(AVG(CASE WHEN lower(CAST(status AS TEXT)) IN ('no_show','no-show','noshow') THEN 1.0 ELSE 0.0 END),3) AS no_show_rate
FROM appointments GROUP BY reminder_sent ORDER BY reminder_sent;

-- Duplicate-key error test (intentional; do not include in the normal query run):
-- CREATE TABLE duplicate_key_test (appointment_id TEXT PRIMARY KEY);
-- INSERT INTO duplicate_key_test VALUES ('A0001');
-- INSERT INTO duplicate_key_test VALUES ('A0001');
-- Expected SQLite result for the second INSERT: UNIQUE constraint failed.
