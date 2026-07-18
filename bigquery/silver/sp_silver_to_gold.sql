CREATE OR REPLACE PROCEDURE `anchoveta.silver.sp_silver_to_gold`()
BEGIN

  -- Gold 1: feature table para modelo (lags/rolling)
  CREATE OR REPLACE TABLE `anchoveta.gold.gold_features` AS
  SELECT *,
    AVG(sst_c) OVER (
      PARTITION BY cell_id ORDER BY fecha
      ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
    ) AS sst_roll4,
    LAG(clorofila_mg_m3) OVER (
      PARTITION BY cell_id ORDER BY fecha
    ) AS chl_lag1
  FROM `anchoveta.silver.silver_anchoveta`;

  -- Gold 2: agregado mensual por zona (BI)
  CREATE OR REPLACE TABLE `anchoveta.gold.gold_agg_zona_mes` AS
  SELECT
    EXTRACT(YEAR FROM fecha) AS anio,
    EXTRACT(MONTH FROM fecha) AS mes,
    zona_pesca,
    AVG(sst_c) AS sst_prom,
    AVG(clorofila_mg_m3) AS chl_prom,
    AVG(densidad_ton_km2) AS densidad_prom,
    COUNT(*) AS n_celdas
  FROM `anchoveta.silver.silver_anchoveta`
  GROUP BY anio, mes, zona_pesca;

END;