## risky asteroids of the week
WEEKLY_RISKY_ASTEROIDS_QUERY = """WITH risky_asteroids_id AS (
    SELECT id
    FROM risk_analysis
    WHERE "RiskCategoryDDriven" = 'High Risk'
    AND risk_analysed_on_date = (
        SELECT MAX(risk_analysed_on_date)
        FROM risk_analysis
    )
)
SELECT name
FROM train_neo
WHERE id IN (SELECT id FROM risky_asteroids_id);"""

SUMMARY_OF_WEEKLY_PREDICTION_QUERY = """WITH weekly_ids AS (
    SELECT id
    FROM risk_analysis
    WHERE risk_analysed_on_date = (
        SELECT MAX(risk_analysed_on_date)
        FROM risk_analysis
    )
)
SELECT
    COUNT(DISTINCT w.id) AS total_asteroids,
    COUNT(DISTINCT w.id) FILTER (WHERE t.is_potentially_hazardous = TRUE) AS hazardous,
    COUNT(DISTINCT w.id) FILTER (WHERE t.is_potentially_hazardous = FALSE) AS non_hazardous
FROM weekly_ids w
LEFT JOIN train_neo t
ON w.id = t.id;
"""

RECIPIENT_DATA_QUERY = "select * from alert_recipients"

SENDER_EMAIL = "aegisaastro@gmail.com"