CREATE OR REPLACE PROCEDURE `anchoveta.raw.sp_raw_to_silver`()
BEGIN

  CREATE OR REPLACE TABLE `anchoveta.silver.s_ubicacion` AS
  SELECT
    cell_id,
    COALESCE(SAFE.PARSE_DATE('%Y-%m-%d', fecha),
             SAFE.PARSE_DATE('%d/%m/%Y', fecha)) AS fecha,
    SAFE_CAST(latitud AS FLOAT64)        AS latitud,
    SAFE_CAST(longitud AS FLOAT64)       AS longitud,
    SAFE_CAST(zona_pesca AS INT64)       AS zona_pesca,
    SAFE_CAST(dist_costa_deg AS FLOAT64) AS dist_costa_deg,
    SAFE_CAST(profundidad_m AS FLOAT64)  AS profundidad_m
  FROM (
    SELECT *, ROW_NUMBER() OVER (
      PARTITION BY cell_id, fecha ORDER BY _ingested_at DESC
    ) AS rn
    FROM `anchoveta.raw.raw_ubicacion`
  )
  WHERE rn = 1;

  CREATE OR REPLACE TABLE `anchoveta.silver.s_oceanografia` AS
  WITH parsed AS (
    SELECT
      cell_id,
      COALESCE(SAFE.PARSE_DATE('%Y-%m-%d', fecha),
               SAFE.PARSE_DATE('%d/%m/%Y', fecha)) AS fecha,
      SAFE_CAST(sst_c AS FLOAT64)           AS sst_c_raw,
      SAFE_CAST(clorofila_mg_m3 AS FLOAT64) AS clorofila_mg_m3,
      SAFE_CAST(salinidad_psu AS FLOAT64)   AS salinidad_psu,
      SAFE_CAST(oxigeno_ml_l AS FLOAT64)    AS oxigeno_ml_l,
      _ingested_at
    FROM `anchoveta.raw.raw_oceanograficas`
  ),
  clean AS (
    SELECT *,
      IF(sst_c_raw < -50, NULL, sst_c_raw) AS sst_c,
      ABS(clorofila_mg_m3) AS chl_abs
    FROM parsed
  ),
  dedup AS (
    SELECT *, ROW_NUMBER() OVER (
      PARTITION BY cell_id, fecha ORDER BY _ingested_at DESC
    ) AS rn
    FROM clean
  ),
  medianas AS (
    SELECT cell_id,
      APPROX_QUANTILES(sst_c, 2)[OFFSET(1)]  AS sst_med,
      APPROX_QUANTILES(chl_abs, 2)[OFFSET(1)] AS chl_med
    FROM dedup WHERE rn = 1
    GROUP BY cell_id
  )
  SELECT d.cell_id, d.fecha,
    LEAST(GREATEST(COALESCE(d.sst_c, m.sst_med), 10), 30) AS sst_c,
    COALESCE(d.chl_abs, m.chl_med) AS clorofila_mg_m3,
    d.salinidad_psu, d.oxigeno_ml_l
  FROM dedup d
  JOIN medianas m USING(cell_id)
  WHERE d.rn = 1 AND d.fecha IS NOT NULL;

  CREATE OR REPLACE TABLE `anchoveta.silver.s_corrientes` AS
  SELECT
    cell_id,
    COALESCE(SAFE.PARSE_DATE('%Y-%m-%d', fecha),
             SAFE.PARSE_DATE('%d/%m/%Y', fecha)) AS fecha,
    SAFE_CAST(corriente_vel_ms AS FLOAT64) AS corriente_vel_ms,
    SAFE_CAST(corriente_u_ms AS FLOAT64)   AS corriente_u_ms,
    SAFE_CAST(corriente_v_ms AS FLOAT64)   AS corriente_v_ms
  FROM (
    SELECT *, ROW_NUMBER() OVER (
      PARTITION BY cell_id, fecha ORDER BY _ingested_at DESC
    ) AS rn
    FROM `anchoveta.raw.raw_corrientes`
  )
  WHERE rn = 1;

  CREATE OR REPLACE TABLE `anchoveta.silver.s_enso` AS
  SELECT fecha, AVG(enso_index) AS enso_index
  FROM (
    SELECT
      COALESCE(SAFE.PARSE_DATE('%Y-%m-%d', fecha),
               SAFE.PARSE_DATE('%d/%m/%Y', fecha)) AS fecha,
      SAFE_CAST(enso_index AS FLOAT64) AS enso_index
    FROM `anchoveta.raw.raw_enso`
  )
  WHERE fecha IS NOT NULL
  GROUP BY fecha;

  CREATE OR REPLACE TABLE `anchoveta.silver.s_biomasa` AS
  SELECT
    cell_id,
    COALESCE(SAFE.PARSE_DATE('%Y-%m-%d', fecha),
             SAFE.PARSE_DATE('%d/%m/%Y', fecha)) AS fecha,
    SAFE_CAST(log_densidad AS FLOAT64)     AS log_densidad,
    SAFE_CAST(densidad_ton_km2 AS FLOAT64) AS densidad_ton_km2
  FROM (
    SELECT *, ROW_NUMBER() OVER (
      PARTITION BY cell_id, fecha ORDER BY _ingested_at DESC
    ) AS rn
    FROM `anchoveta.raw.raw_biomasa`
  )
  WHERE rn = 1;

  CREATE OR REPLACE TABLE `anchoveta.silver.silver_anchoveta` AS
  SELECT
    u.cell_id, u.fecha,
    EXTRACT(YEAR FROM u.fecha)    AS anio,
    EXTRACT(MONTH FROM u.fecha)   AS mes,
    EXTRACT(ISOWEEK FROM u.fecha) AS semana_anio,
    u.latitud, u.longitud, u.zona_pesca,
    u.dist_costa_deg, u.profundidad_m,
    o.sst_c, o.clorofila_mg_m3, o.salinidad_psu, o.oxigeno_ml_l,
    c.corriente_vel_ms, c.corriente_u_ms, c.corriente_v_ms,
    e.enso_index,
    b.log_densidad, b.densidad_ton_km2
  FROM `anchoveta.silver.s_ubicacion` u
  JOIN `anchoveta.silver.s_oceanografia` o USING(cell_id, fecha)
  JOIN `anchoveta.silver.s_corrientes`   c USING(cell_id, fecha)
  JOIN `anchoveta.silver.s_biomasa`      b USING(cell_id, fecha)
  LEFT JOIN `anchoveta.silver.s_enso`    e USING(fecha);

END;