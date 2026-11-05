# AirCast Research: Exploratory Data Analysis & Statistical Synthesis

**Document Version:** 1.0.0  
**Dataset:** Beijing Multi-Site Air Quality Dataset (PRSA 2017)  
**Sample Period:** 2013-03-01 00:00:00 to 2017-02-28 23:00:00 UTC+8  
**Total Observations:** 420,768 hourly records across 12 monitoring stations  
**Generated Artifacts:** 
- Visualizations: [artifacts/plots/](../artifacts/plots)
- Statistical Summary: [artifacts/metrics/eda_statistical_summary.json](../artifacts/metrics/eda_statistical_summary.json)

---

## 1. Executive Summary & Research Context

AirCast Research formulates particulate matter ($PM_{2.5}$) forecasting as a multivariate, multi-horizon time-series regression problem across 12 spatially distributed monitoring stations in Beijing. Understanding the statistical properties of the telemetry, underlying physical atmospheric mechanisms, spatial heterogeneity, and data topology is a prerequisite for developing robust, generalizable predictive models.

This report documents the empirical findings from analyzing 420,768 hourly observations across six criteria air pollutants ($PM_{2.5}, PM_{10}, SO_2, NO_2, CO, O_3$) and five surface meteorological variables ($TEMP, PRES, DEWP, RAIN, WSPM, wd$). All statistics and figures presented herein were derived directly from the provided raw files without modification.

---

## 2. Univariate Distributional Properties of Criteria Pollutants

Atmospheric pollutant concentrations exhibit extreme positive skewness and heavy right tails, driven by episodic synoptic stagnation events and localized emission bursts.

![Pollutant Distributions](../artifacts/plots/eda_pollutant_distributions.png)

### Summary Statistics and Non-Normality Diagnostics

| Pollutant | Valid N | Mean ($\mu\text{g}/\text{m}^3$) | Std ($\mu\text{g}/\text{m}^3$) | Median | IQR | Min | Max | Skewness ($\gamma_1$) | Excess Kurtosis ($\gamma_2$) | Jarque-Bera $p$-value |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **$PM_{2.5}$** | 412,029 | 79.79 | 80.82 | 55.00 | 91.00 | 2.00 | 999.00 | +2.014 | +5.965 | $< 10^{-15}$ |
| **$PM_{10}$** | 414,319 | 104.60 | 91.77 | 82.00 | 109.00 | 2.00 | 999.00 | +1.886 | +6.186 | $< 10^{-15}$ |
| **$SO_2$** | 411,747 | 15.83 | 21.65 | 7.00 | 17.00 | 0.29 | 500.00 | +3.009 | +14.188 | $< 10^{-15}$ |
| **$NO_2$** | 408,652 | 50.64 | 35.13 | 43.00 | 48.00 | 1.03 | 290.00 | +1.050 | +1.204 | $< 10^{-15}$ |
| **$CO$** | 400,067 | 1230.77 | 1160.18 | 900.00 | 1000.00 | 100.00 | 10000.00 | +2.570 | +9.319 | $< 10^{-15}$ |
| **$O_3$** | 407,491 | 57.37 | 56.66 | 45.00 | 71.00 | 0.21 | 1071.00 | +1.660 | +6.264 | $< 10^{-15}$ |

### Scientific Insights
1. **Severe Positive Skewness:** For $PM_{2.5}$, the sample mean ($79.79\ \mu\text{g}/\text{m}^3$) substantially exceeds the median ($55.00\ \mu\text{g}/\text{m}^3$), with a skewness of $+2.014$. Least-squares estimators (e.g., standard OLS) minimize quadratic loss ($\mathbb{E}[(y - \hat{y})^2]$) and are notoriously sensitive to extreme values. Models using robust loss functions (e.g., Huber loss, pinball loss for quantile forecasting, or tree-based splits optimizing MAE/MSE) and target transformations must be systematically evaluated.
2. **Heavy Kurtosis:** $SO_2$ displays an extreme excess kurtosis of $+14.188$, characterized by near-zero background levels during summer punctuated by massive winter peaks ($> 200\ \mu\text{g}/\text{m}^3$) due to centralized coal-fired district heating.
3. **Rejection of Normality:** Jarque-Bera test statistics are astronomically high ($p < 10^{-15}$ across all pollutants), confirming that standard Gaussian assumptions are completely violated.

---

## 3. Temporal Dynamics: Diurnal, Day-of-Week, and Seasonal Cycles

The concentration of $PM_{2.5}$ is governed by interactions between anthropogenic emissions and the diurnal evolution of the atmospheric planetary boundary layer (PBL).

![Temporal and Seasonal Variations](../artifacts/plots/eda_temporal_seasonal_patterns.png)

### Diurnal Evolution and Boundary Layer Physics
- **Nocturnal Accumulation:** Concentrations peak between 21:00 and 02:00 local time (exceeding $95\ \mu\text{g}/\text{m}^3$ during winter heating periods). This is caused by radiative surface cooling that generates a shallow thermal inversion layer (confining pollutants to the lowest $100\text{–}300\ \text{meters}$ of the atmosphere), coupled with the night-time influx of heavy-duty diesel freight trucks permitted into Beijing only after 20:00.
- **Midday Convective Cleansing:** A sharp diurnal trough occurs between 13:00 and 16:00 local time (reaching daily lows of $\approx 50\ \mu\text{g}/\text{m}^3$). Solar heating drives vertical convective turbulent mixing, lifting the PBL height to $1,500\text{–}2,000\ \text{meters}$ and diluting ground-level particulate mass.

### Seasonal Stratification: Heating vs. Non-Heating
- **Heating Season (Nov 15 – Mar 15):** Mean $PM_{2.5} = 94.88\ \mu\text{g}/\text{m}^3$.
- **Non-Heating Season (Mar 16 – Nov 14):** Mean $PM_{2.5} = 69.05\ \mu\text{g}/\text{m}^3$.
- **Heating Amplification Ratio:** $1.37\times$ ($+37.4\%$ higher pollution during the heating regime). This divergence is driven by two reinforcing factors:
  1. Primary coal burning for residential heating in the surrounding Hebei-Shandong-Henan industrial belt.
  2. Synoptic high-pressure stagnation systems and intense temperature inversions common during Northern China winters.

### Day-of-Week Patterns
- **Weekday Mean:** $78.08\ \mu\text{g}/\text{m}^3$
- **Weekend Mean:** $84.04\ \mu\text{g}/\text{m}^3$ ($+7.6\%$ higher on weekends).
- This "weekend effect" is an established urban phenomenon in Beijing: industrial and commercial transport activities ramp up on Friday night and Saturday, while atmospheric stagnation causes pollutant buildup over 48–72 hour accumulation cycles.

---

## 4. Spatial Heterogeneity and Cross-Station Pollution Gradients

The 12 monitoring stations span distinct geographical, topography-bounded microclimates across the Beijing metropolitan basin.

![Station Comparison](../artifacts/plots/eda_station_comparison.png)

### Station-Level Pollution Distribution (Sorted by Median)

| Station Identifier | Station Classification | Sample Count | Mean ($\mu\text{g}/\text{m}^3$) | Std ($\mu\text{g}/\text{m}^3$) | Median ($\mu\text{g}/\text{m}^3$) | IQR ($\mu\text{g}/\text{m}^3$) |
| :--- | :--- | :-: | :-: | :-: | :-: | :-: |
| **Dingling** | Regional Background (Ming Tombs) | 34,285 | 65.99 | 72.27 | **41.00** | 79.00 |
| **Changping** | Suburban Northwest | 34,290 | 71.10 | 72.33 | **46.00** | 82.00 |
| **Huairou** | Mountain Buffer (North) | 34,111 | 69.63 | 71.22 | **47.00** | 81.00 |
| **Shunyi** | Suburban Northeast (Airport) | 34,151 | 79.49 | 81.23 | **55.00** | 93.00 |
| **Aotizhongxin** | Urban Core (Olympic Park) | 34,139 | 82.77 | 82.14 | **58.00** | 92.00 |
| **Nongzhanguan**| Urban Core (East 3rd Ring) | 34,436 | 84.84 | 86.23 | **59.00** | 94.00 |
| **Tiantan** | Urban Core (Temple of Heaven) | 34,387 | 82.16 | 80.92 | **59.00** | 91.00 |
| **Guanyuan** | Urban Core (Xicheng) | 34,448 | 82.93 | 80.93 | **59.00** | 92.00 |
| **Wanliu** | Urban Tech Corridor (Haidian) | 34,682 | 83.37 | 81.91 | **59.00** | 93.00 |
| **Gucheng** | Industrial Urban (Shijingshan) | 34,418 | 83.85 | 82.80 | **60.00** | 91.75 |
| **Wanshouxigong**| Urban Core (South 2nd Ring) | 34,368 | 85.02 | 85.98 | **60.00** | 93.00 |
| **Dongsi** | Urban Core (Dongcheng) | 34,314 | 86.19 | 86.58 | **61.00** | 97.00 |

### Spatial Insights for Generalisation (RQ4)
1. **Background Clean Baseline:** Dingling (located near the northern mountain ecological reserve) has a median $PM_{2.5}$ of $41.00\ \mu\text{g}/\text{m}^3$ and mean of $65.99\ \mu\text{g}/\text{m}^3$—by far the cleanest site in the network.
2. **South-to-North Pollution Gradient:** Southern and central stations (Dongsi, Wanshouxigong, Nongzhanguan, Gucheng) exhibit median concentrations of $59\text{–}61\ \mu\text{g}/\text{m}^3$ ($+48.8\%$ above Dingling). This reflects Beijing's topographic bowl effect: pollution transported from the southern Hebei industrial corridor is funneled into the southern urban core and trapped against the Yan Mountains to the north.
3. **Leave-One-Station-Out Sensitivity:** Evaluating models on held-out stations will stress-test whether algorithms learn generalizable meteorological-dispersion physics or overfit to station-specific baseline offsets.

---

## 5. Meteorological Drivers: Dispersion and Atmospheric Chemistry

Meteorological conditions exert first-order controls on particulate concentration, governing physical dispersion, chemical gas-to-particle conversion, and deposition.

![Meteorological Relationships](../artifacts/plots/eda_meteorological_relationships.png)

### Key Meteorological Mechanisms

1. **Wind Speed Dispersion Threshold:**
   - At calm wind speeds ($WSPM \le 1.0\ \text{m}/\text{s}$), median $PM_{2.5}$ exceeds $85\ \mu\text{g}/\text{m}^3$.
   - A distinct physical transition occurs at **$2.0\ \text{m}/\text{s}$**: beyond this ventilation threshold, advective ventilation dominates, driving median $PM_{2.5}$ precipitously down to $< 25\ \mu\text{g}/\text{m}^3$ at $WSPM \ge 4.0\ \text{m}/\text{s}$.
   - Linear correlation: $r = -0.272$; Non-linear rank correlation: $\rho = -0.329$.

2. **Wind Direction Sector Bimodality:**
   - **Clean Northerly Winds ($N, NNW, NW$):** Siberian anticyclonic fronts sweep fast, cold, dry air over the Yan Mountains, purging the Beijing basin (mean $PM_{2.5} = 42\text{–}48\ \mu\text{g}/\text{m}^3$).
   - **Polluted Southerly Winds ($S, SSW, SE, ESE$):** Slow-moving maritime air masses transport heavy precursor emissions from the southern Hebei industrial basin, causing severe haze accumulation (mean $PM_{2.5} = 98\text{–}108\ \mu\text{g}/\text{m}^3$).

3. **Dew Point Depression ($TEMP - DEWP$) & Secondary Aerosol Formation:**
   - Dew point depression serves as a physical proxy for relative humidity ($RH \approx 100 - 5 \times [TEMP - DEWP]$).
   - Spearman rank correlation: **$\rho = -0.459$** (one of the strongest meteorological predictors).
   - Under moist conditions ($TEMP - DEWP \le 3^\circ\text{C}$), median $PM_{2.5}$ surges above $110\ \mu\text{g}/\text{m}^3$. High relative humidity accelerates aqueous-phase secondary inorganic aerosol (SIA) synthesis: gaseous $SO_2$ and $NO_2$ rapidly convert to ammonium sulfate $((NH_4)_2SO_4)$ and ammonium nitrate $(NH_4NO_3)$ within hygroscopic water droplets.

4. **Temperature Dynamics:**
   - Low temperatures ($\le -5^\circ\text{C}$) correlate with high $PM_{2.5}$ (winter heating and thermal inversions).
   - Moderate temperatures ($15\text{–}22^\circ\text{C}$) in spring/autumn correlate with lower pollution.
   - High temperatures in summer ($> 30^\circ\text{C}$) correspond to peak $O_3$ photochemistry.

---

## 6. Correlation Analysis & Multicollinearity Assessment

The correlation matrix reveals pronounced linear (Pearson) and monotonic (Spearman) dependencies among co-emitted pollutants and meteorological drivers.

![Correlation Matrix](../artifacts/plots/eda_correlation_matrix.png)

### Correlation Matrix with Primary Target ($PM_{2.5}$)

| Feature | Pearson Correlation ($r$) | Spearman Rank Correlation ($\rho$) | Physical / Atmospheric Mechanism |
| :--- | :-: | :-: | :--- |
| **$PM_{10}$** | **+0.884** | **+0.891** | Shared particulate emission sources; $PM_{2.5}$ is a subcomponent of $PM_{10}$. |
| **$CO$** | **+0.790** | **+0.837** | Tracer of incomplete fossil fuel and biomass combustion; unreactive over short time horizons. |
| **$NO_2$** | **+0.667** | **+0.658** | Primary vehicle and urban combustion emissions; nitrate aerosol precursor. |
| **$SO_2$** | **+0.482** | **+0.500** | Coal burning marker; precursor to secondary sulfate particulate mass. |
| **Dew Point Depression**| **-0.391** | **-0.459** | Moisture indicator ($TEMP - DEWP$); low depression = high humidity = secondary aqueous haze. |
| **$WSPM$** (Wind Speed)| **-0.272** | **-0.329** | Advective atmospheric ventilation and turbulent dilution. |
| **$O_3$** | **-0.150** | **-0.254** | Secondary photochemical titration; negatively correlated in winter, out of phase with haze. |
| **$TEMP$** | **-0.131** | **-0.024** | Seasonal temperature proxy (winter heating vs summer dilution). |
| **$RAIN$** | **-0.014** | **-0.020** | Wet atmospheric washout (rainfall is rare, 95.98% dry hours). |
| **$PRES$** | **+0.019** | **-0.070** | Synoptic surface barometric pressure. |

### Multicollinearity Deductions
- **Severe Collinearity Between $PM_{2.5}, PM_{10}$, and $CO$:** The Spearman correlation between $PM_{2.5}$ and $CO$ is $+0.837$, and with $PM_{10}$ is $+0.891$. In linear regression models, simultaneous inclusion of raw $PM_{10}$ and $CO$ without regularisation (Ridge/Lasso) will inflate coefficient variance (high Variance Inflation Factor, $\text{VIF} > 10$). Tree-based models (Random Forest, XGBoost, LightGBM) natively handle collinearity by splitting on correlated features, but feature ablation (Experiment 3) is necessary to measure true marginal importance.

---

## 7. Topology of Unobserved Data: Streaks and Imputation Feasibility

Understanding the distribution of unobserved periods informs the design of station-aware causal imputation pipelines.

![Missingness Patterns](../artifacts/plots/eda_missingness_patterns.png)

### Missingness Properties
- **Total Unobserved $PM_{2.5}$:** **8,739 observations (2.0769%)**.
- **Consecutive Streak Lengths:**
  - **Median Streak Length:** **1.0 hour**.
  - **75th Percentile Streak Length:** **2.0 hours**.
  - **90th Percentile Streak Length:** **4.0 hours**.
  - **Short Gap Proportion ($\le 3$ hours):** **87.2%** of all missing streaks.
  - **Maximum Streak Length:** 343 hours ($\approx 14$ days, occurring during a major sensor maintenance outage at Huairou).

### Methodological Implications
Because **87.2% of unobserved gaps are 3 hours or shorter**, station-aware causal forward-filling ($\text{ffill}$) captures persistent atmospheric states effectively with minimal distortion. For prolonged multi-day outages ($> 72$ hours), forward-filling stale measurements degrades fidelity; hence, a 72-hour capping threshold falling back to training partition medians, alongside binary missingness indicators (`PM2.5_missing`), is implemented in `StationAwareCausalImputer`.

---

## 8. In-Depth Investigation: $PM_{2.5} > PM_{10}$ Discrepancy

Under ideal physical particle sizing, $PM_{2.5}$ is a subcomponent of $PM_{10}$ ($PM_{2.5} \subset PM_{10}$), implying $PM_{2.5} \le PM_{10}$.

![PM Ratio Discrepancy](../artifacts/plots/eda_pm_ratio_discrepancy.png)

### Empirical Findings
- **Total Valid Simultaneous Measurement Pairs (Denominator):** **411,813 pairs** (where $PM_{2.5} \neq \text{NaN} \land PM_{10} \neq \text{NaN}$).
- **Count where $PM_{2.5} > PM_{10}$:** **21,312 pairs**.
- **Exact Percentage of Valid Pairs:** **5.175164%** ($5.065024\%$ of all 420,768 records).
- **Discrepancy Magnitude Distribution ($\Delta = PM_{2.5} - PM_{10}$):**
  - **Mean $\Delta$:** $24.72\ \mu\text{g}/\text{m}^3$
  - **Median $\Delta$:** $19.00\ \mu\text{g}/\text{m}^3$
  - **25th Percentile:** $8.00\ \mu\text{g}/\text{m}^3$
  - **75th Percentile:** $34.00\ \mu\text{g}/\text{m}^3$
  - **95th Percentile:** $66.00\ \mu\text{g}/\text{m}^3$
  - **Maximum $\Delta$:** $591.00\ \mu\text{g}/\text{m}^3$ (recorded at Dingling, 2014-04-12 17:00).
- **Station Breakdown:**
  - Dingling has the highest inversion count: **2,949 instances** ($13.8\%$ of all inversions).
  - Dongsi: **2,156 instances**.
  - Aotizhongxin: **1,183 instances** (lowest).

### Physical Attribution & Scientific Protocol
1. **Instrument Channel Independence:** Continuous ambient monitoring stations operate distinct instrument sampling heads for $PM_{2.5}$ and $PM_{10}$ (e.g., separate $PM_{2.5}$ cyclone/impactors vs $PM_{10}$ heads on Beta Attenuation Monitors [BAM] or TEOM systems).
2. **Hygroscopic Condensation:** During rapid surges in relative humidity, deliquescent inorganic salts (sulfates/nitrates) swell within the $PM_{2.5}$ inlet or optical scattering chamber at differing rates, producing temporary instrument divergence.
3. **Actionable Protocol:** In accordance with scientific integrity guidelines, these observations are **preserved without alteration or deletion**. They represent legitimate historical telemetry outputs with documented sensor noise.

---

## 9. Extreme Pollution Events & Air Quality Categorization

Beijing experiences severe episodic winter haze events that pose acute public health risks. Evaluating models across these extreme distribution regimes is the focus of Experiment 8.

![Extreme Pollution Events](../artifacts/plots/eda_extreme_pollution_events.png)

### US EPA Air Quality Index (AQI) Stratification

| Health Category | $PM_{2.5}$ Range ($\mu\text{g}/\text{m}^3$) | Observation Count | Percentage of Valid | Cumulative (%) |
| :--- | :--- | :-: | :-: | :-: |
| **Good** | $0.0 \text{ to } 12.0$ | 62,775 | 15.24% | 15.24% |
| **Moderate** | $12.1 \text{ to } 35.4$ | 91,096 | 22.11% | 37.35% |
| **Unhealthy for Sensitive Groups** | $35.5 \text{ to } 55.4$ | 53,248 | 12.92% | 50.27% |
| **Unhealthy** | $55.5 \text{ to } 150.4$ | 142,240 | 34.52% | 84.79% |
| **Very Unhealthy** | $150.5 \text{ to } 250.4$ | 44,234 | 10.74% | 95.53% |
| **Hazardous** | $> 250.4$ | **18,436** | **4.47%** | **100.00%** |

### Severe Haze Analysis
- **Hazardous Exposure:** Over **18,436 hours (4.47%)** fall into the Hazardous category ($PM_{2.5} > 250.4\ \mu\text{g}/\text{m}^3$).
- **Extreme Peak Events ($> 500\ \mu\text{g}/\text{m}^3$):** **933 hours** exceed $500\ \mu\text{g}/\text{m}^3$.
- **Case Study: The Historic December 2015 "Red Alert":**
  - Between November 27 and December 10, 2015, Beijing issued its first-ever official municipal "Red Alert" for air pollution.
  - The network-wide station mean exceeded $350\ \mu\text{g}/\text{m}^3$ continuously for over 96 hours, with individual stations reaching $999.0\ \mu\text{g}/\text{m}^3$ (instrument saturation ceiling).
  - These events are characterized by extreme temporal autocorrelation (persistence) and sudden clearing once a Northerly cold front arrives. Models must handle both the rapid onset and precipitous collapse of these episodes.

---

## 10. Summary of Key Analytical Decisions for Phases 3–5

1. **Target Formulation:** Multi-horizon forecasting ($h \in \{1, 6, 24\}$ hours). Unobserved targets are strictly filtered out; never synthetically imputed.
2. **Feature Imputation:** Station-aware causal forward-fill with a 72-hour limit and training median fallback, augmented with binary missing indicators.
3. **Temporal Partitioning:** Chronological splits with an embargo buffer $\ge 24$ hours between train, validation, and test to guarantee zero target leakage.
4. **Spatial Partitioning:** Leave-one-station-out cross-station generalisation with complete isolation of the held-out station during fitting.
5. **Feature Engineering Focus:** Cyclical temporal encoding ($\sin/\cos$ for hour/month), circular wind vector decomposition ($u, v$), non-linear moisture proxies (dew point depression), and station-level lag/rolling dynamics.
