# AirCast Research: Beijing Multi-Site Air Quality Data Dictionary

## 1. Dataset Overview

- **Dataset Identifier:** Beijing Multi-Site Air Quality Dataset (PRSA 2017)
- **Primary Source:** Beijing Municipal Environmental Monitoring Center / Tsinghua University
- **Period of Record:** 2013-03-01 00:00:00 to 2017-02-28 23:00:00 UTC+8 (Beijing Local Time)
- **Total Duration:** 1,461 calendar days (35,064 consecutive hours per station, including the 2016 leap year)
- **Spatial Coverage:** 12 air quality monitoring stations in the Beijing Metropolitan Area
- **Total Observations:** 420,768 hourly records (12 stations × 35,064 hours)
- **Raw File Format:** 12 individual Comma-Separated Values (`.csv`) files inside `PRSA_Data_20130301-20170228/`
- **File Dimensions:** Exactly 35,064 rows × 18 columns per station file

---

## 2. Monitoring Stations Metadata

| Station Name | Chinese Name | Station Classification | Geographical / Environmental Context | Latitude / Longitude (approx.) | Unobserved PM2.5 (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Aotizhongxin** | 奥体中心 | Urban Core (North) | Olympic Sports Center, North 4th Ring Road, heavy traffic & urban park mix | 39.982° N, 116.397° E | 2.64% (925 rows) |
| **Changping** | 昌平 | Suburban (Northwest) | Changping town center, transition between urban basin and Yan Mountains | 40.217° N, 116.230° E | 2.21% (774 rows) |
| **Dingling** | 定陵 | Regional Background | Ming Tombs ecological reserve, baseline reference site with minimal local emissions | 40.292° N, 116.220° E | 2.22% (779 rows) |
| **Dongsi** | 东四 | Urban Core (Central) | Dongcheng District, dense residential, commercial, and traditional courtyard zone | 39.929° N, 116.417° E | 2.14% (750 rows) |
| **Guanyuan** | 官园 | Urban Core (West) | Xicheng District, residential area with educational and administrative institutions | 39.929° N, 116.339° E | 1.76% (616 rows) |
| **Gucheng** | 古城 | Industrial / Urban (West) | Shijingshan District, former heavy metallurgical industrial corridor (Capital Steel) | 39.914° N, 116.184° E | 1.84% (646 rows) |
| **Huairou** | 怀柔 | Suburban / Rural (North) | Huairou reservoir basin, northern mountain barrier, seasonal recreation | 40.328° N, 116.628° E | 2.72% (953 rows) |
| **Nongzhanguan**| 农展馆 | Urban Core (East) | Chaoyang District, National Agricultural Exhibition Center, East 3rd Ring | 39.937° N, 116.461° E | 1.79% (628 rows) |
| **Shunyi** | 顺义 | Suburban (Northeast) | Shunyi District, airport industrial corridor, low-density residential mix | 40.127° N, 116.655° E | 2.60% (913 rows) |
| **Tiantan** | 天坛 | Urban Core (South) | Dongcheng District, Temple of Heaven historical park, southern inner city | 39.886° N, 116.407° E | 1.93% (677 rows) |
| **Wanliu** | 万柳 | Urban / Educational (NW) | Haidian District, Zhongguancun tech corridor, high-density residential | 39.987° N, 116.287° E | 1.09% (382 rows) |
| **Wanshouxigong**| 万寿西宫 | Urban Core (SW) | Xicheng District, Baizhifang area, dense urban residential, South 2nd Ring | 39.878° N, 116.352° E | 1.98% (696 rows) |

---

## 3. Comprehensive Column Data Dictionary

| Column Name | Inferred Type | Pandas Dtype | Physical Unit | Minimum | Maximum | Mean ± Std | Unobserved Count (%) | Physical Description & Analytical Role |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `No` | Discrete Index | `int64` | Dimensionless | 1 | 35064 | 17532.5 ± 10122.2 | 0 (0.00%) | Sequential record index per station file. Strictly monotonic from 1 to 35,064. |
| `year` | Temporal Discrete | `int64` | Calendar Year | 2013 | 2017 | 2014.66 ± 1.18 | 0 (0.00%) | Calendar year (2013 to 2017). |
| `month` | Temporal Discrete | `int64` | Calendar Month | 1 | 12 | 6.52 ± 3.45 | 0 (0.00%) | Calendar month (1 to 12). Strong indicator for seasonal heating vs. non-heating regimes. |
| `day` | Temporal Discrete | `int64` | Calendar Day | 1 | 31 | 15.73 ± 8.80 | 0 (0.00%) | Day of month (1 to 31). |
| `hour` | Temporal Discrete | `int64` | Hour of Day | 0 | 23 | 11.50 ± 6.92 | 0 (0.00%) | Hour of day (0 to 23, Beijing Time). Key driver of diurnal traffic and atmospheric boundary layer height. |
| `PM2.5` | Continuous Float | `float64` | $\mu\text{g}/\text{m}^3$ | 2.0 | 999.0 | 79.79 ± 80.82 | 8,739 (2.08%) | **Primary Target Variable:** Fine particulate matter ($\le 2.5\ \mu\text{m}$). Right-skewed distribution; sensor clipping ceiling at 999 $\mu\text{g}/\text{m}^3$. Exclude unobserved targets from training/evaluation. |
| `PM10` | Continuous Float | `float64` | $\mu\text{g}/\text{m}^3$ | 2.0 | 999.0 | 104.60 ± 91.77 | 6,449 (1.53%) | Inhalable particulate matter ($\le 10\ \mu\text{m}$). Includes coarse dust, road dust, and fine particles. |
| `SO2` | Continuous Float | `float64` | $\mu\text{g}/\text{m}^3$ | 0.29 | 500.0 | 15.83 ± 21.65 | 9,021 (2.14%) | Sulfur dioxide. Primary marker for coal combustion and winter residential heating. |
| `NO2` | Continuous Float | `float64` | $\mu\text{g}/\text{m}^3$ | 1.03 | 290.0 | 50.64 ± 35.13 | 12,116 (2.88%) | Nitrogen dioxide. Primary marker for vehicular emissions and urban fossil fuel burning. |
| `CO` | Continuous Float | `float64` | $\mu\text{g}/\text{m}^3$ | 100.0 | 10000.0 | 1230.77 ± 1160.18 | 20,701 (4.92%) | Carbon monoxide. Incomplete combustion tracer. Detection lower threshold: 100, clipping upper limit: 10,000. |
| `O3` | Continuous Float | `float64` | $\mu\text{g}/\text{m}^3$ | 0.21 | 1071.0 | 57.37 ± 56.66 | 13,277 (3.16%) | Ground-level tropospheric ozone. Secondary photochemical oxidant; negatively correlated with PM2.5 in winter, positively in summer. |
| `TEMP` | Continuous Float | `float64` | $^\circ\text{C}$ | -19.9 | 41.6 | 13.54 ± 11.44 | 398 (0.09%) | Ambient surface dry-bulb air temperature. Controls chemical reaction rates and boundary layer inversion dynamics. |
| `PRES` | Continuous Float | `float64` | $\text{hPa}$ | 982.4 | 1042.8 | 1010.75 ± 10.47 | 393 (0.09%) | Surface barometric pressure. Tracks synoptic meteorological pressure systems (Siberian High vs. Mongolian Low). |
| `DEWP` | Continuous Float | `float64` | $^\circ\text{C}$ | -43.4 | 29.1 | 2.49 ± 13.79 | 403 (0.10%) | Dew point temperature. Relative humidity indicator ($TEMP - DEWP$); moisture accelerates secondary aerosol aqueous formation. |
| `RAIN` | Continuous Float | `float64` | $\text{mm}$ | 0.0 | 72.5 | 0.06 ± 0.82 | 390 (0.09%) | Accumulated hourly precipitation. Atmospheric wet scavenging/washout mechanism. 95.98% genuine physical zeros. |
| `wd` | Nominal Categorical | `object` / `str` | Compass Point | N/A | N/A | Mode: NE (43,335) | 1,822 (0.43%) | 16-point compass wind direction (N, NNE, NE, ENE, E, ESE, SE, SSE, S, SSW, SW, WSW, W, WNW, NW, NNW). Encoded cyclically ($\sin/\cos$) in feature pipeline. |
| `WSPM` | Continuous Float | `float64` | $\text{m}/\text{s}$ | 0.0 | 13.2 | 1.73 ± 1.25 | 318 (0.08%) | Wind speed at 10m height. Main atmospheric dispersion factor; low wind speeds ($\le 1.5\ \text{m}/\text{s}$) correlate strongly with haze accumulation. |
| `station` | Nominal Categorical | `object` / `str` | Name Identifier | N/A | N/A | Balanced (35,064 per station) | 0 (0.00%) | Name of the monitoring station. Used for station-specific grouping, stratified feature engineering, and cross-station spatial generalisation. |

---

## 4. Key Physical, Methodological, and Data Quality Principles

1. **Station Continuity & Monotonic Time Series:**
   - Every station file contains an unbroken hourly time sequence spanning 35,064 hours without any missing timestamp rows.
   - All files have identical column layouts and schemas.

2. **Absence of Zero Pollutant Readings:**
   - There are **zero** instances of `0.0` across all six pollutants (`PM2.5`, `PM10`, `SO2`, `NO2`, `CO`, `O3`).
   - Pollutant non-detections are recorded either at instrument-specific detection limits (`PM2.5` min = 2.0 $\mu\text{g}/\text{m}^3$, `CO` min = 100 $\mu\text{g}/\text{m}^3$) or as unobserved entries (`NaN`).
   - Genuine physical zeros occur exclusively in `RAIN` (95.98%) and `WSPM` (2.64%).
   - Missing target observations (`PM2.5(t+h)`) must **never** be imputed. Samples lacking target values are cleanly dropped from supervised modeling.
   - For historical input features, station-aware causal forward imputation is applied, accompanied by missingness indicators.

3. **Rigorous Verification of $PM_{2.5} > PM_{10}$ Observations:**
   - Valid simultaneous measurement pairs ($PM_{2.5} \neq \text{NaN} \land PM_{10} \neq \text{NaN}$): **411,813 pairs**.
   - Count where $PM_{2.5} > PM_{10}$: **21,312 pairs**.
   - Exact percentage of valid simultaneous pairs: **5.175164%** (or $5.065024\%$ of all 420,768 records).
   - Mean discrepancy: $24.72\ \mu\text{g}/\text{m}^3$, median: $19.00\ \mu\text{g}/\text{m}^3$, maximum: $591.00\ \mu\text{g}/\text{m}^3$.
   - These observations are preserved without arbitrary alteration or deletion.

4. **Extreme Pollution Events vs. Sensor Saturation:**
   - Maximum values for $PM_{2.5}$ (999.0 $\mu\text{g}/\text{m}^3$) and $PM_{10}$ (999.0 $\mu\text{g}/\text{m}^3$) reflect instrument upper detection limits reached during severe winter regional haze episodes.
   - Only 1 record reaches exactly 999.0 for PM2.5, and 3 are $\ge 900.0$, forming a smooth continuum from 800 to 999 $\mu\text{g}/\text{m}^3$. They represent true environmental episodes and are preserved for extreme-event evaluation.

5. **Strict Temporal Partitioning & Boundary Purge:**
   - Chronological split: Training (2013-03-01 to 2015-12-31), Validation (2016-01-01 to 2016-09-30), Testing (2016-10-01 to 2017-02-28).
   - A **24-hour boundary purge (embargo buffer)** is enforced between partitions so that targets up to $h=24$ hours cannot cross from training into validation, or validation into test.
   - All transformers, scalers, and imputers are fitted solely on the training partition.
