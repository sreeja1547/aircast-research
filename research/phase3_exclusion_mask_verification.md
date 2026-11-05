# AirCast Research: Final Exclusion Mask & Set-Theoretic Accounting Verification

**Document Version:** 1.0.0  
**Verification Date:** November 2026  
**Auditor:** Senior Applied Scientist & Mathematical Reviewer  
**Scope:** Mathematical proof and empirical verification of sample exclusion masks and set unions across all 12 stations and 3 forecasting horizons.  
**Machine-Readable Summary Artifact:** [artifacts/metrics/full_dataset_integration_metrics.json](../artifacts/metrics/full_dataset_integration_metrics.json)  

---

## 1. Mathematical Framework & Objective

In multi-horizon time-series dataset preparation, supervised instances $(\mathbf{X}(t), y_s^{(h)}(t))$ are constructed by aligning historical features $\mathbf{X}(t)$ with future raw targets $y_s^{(h)}(t) = PM_{2.5, s}(t + h)$. 

A sample at timestamp $t$ for station $s$ is excluded if and only if it satisfies at least one exclusion condition:
1. **Horizon-Boundary Set ($B$):** The timestamp $t$ falls within the final $h$ hours of the station series, meaning $t + h$ lies beyond the station timeline:
   $$B = \{ t : t \ge T_s - h \}$$
2. **Missing Target Set ($M_h$):** The raw target $PM_{2.5, s}(t + h)$ was unobserved in the sensor telemetry ($t < T_s - h \land PM_{2.5, s}(t + h) = \text{NaN}$):
   $$M_h = \{ t : t < T_s - h \land y_s^{\text{raw}}(t + h) = \text{NaN} \}$$
3. **Lag Warmup Set ($W$):** The timestamp $t$ falls within the initial 48 hours of the station series ($t < 48$), meaning the 48-hour autoregressive lag ($PM_{2.5\_lag\_48}$) cannot be evaluated:
   $$W = \{ t : t < 48 \}$$

### Set-Theoretic Union Formula
By the Principle of Inclusion-Exclusion, the total number of excluded rows is:
$$|B \cup M_h \cup W| = |B| + |M_h| + |W| - (|B \cap M_h| + |B \cap W| + |M_h \cap W|) + |B \cap M_h \cap W|$$

The final usable sample count is:
$$N_{\text{usable}} = N_{\text{total}} - |B \cup M_h \cup W|$$

---

## 2. Answers to the Six Specific Review Inquiries

### Check 1: Do missing raw PM2.5 observations occur in the final $h$ hours of each station?
- **For $h=1$:** The boundary set $B$ consists of the single final hour of each station: `2017-02-28 23:00:00`.
  - In all 12 stations, the raw measurement at `2017-02-28 23:00:00` was **fully observed** ($0$ NaNs).
- **For $h=6$:** The boundary set $B$ consists of the final 6 hours (`2017-02-28 18:00:00` to `23:00:00`).
  - In all 12 stations, raw $PM_{2.5}$ measurements throughout these final 6 hours were **fully observed** ($0$ NaNs).
- **For $h=24$:** The boundary set $B$ consists of the final 24 hours (`2017-02-28 00:00:00` to `23:00:00`).
  - Raw missing observations did occur during this 24-hour period at two stations: **Guanyuan** (8 missing hours: 00:00 to 07:00) and **Nongzhanguan** (3 missing hours: 02:00, 14:00, 15:00).
  - **Does this cause an overlap between $B$ and $M_{24}$?**
    - **No.** In direct forecasting, a missing raw value at timestamp $\tau$ serves as a target for row $t = \tau - 24$.
    - Because $\tau \in [T_s - 24, T_s - 1]$, the corresponding prediction timestamp is $t = \tau - 24 \in [T_s - 48, T_s - 25]$.
    - Meanwhile, boundary rows $B$ occupy timestamps $t \in [T_s - 24, T_s - 1]$.
    - Because $[T_s - 48, T_s - 25] \cap [T_s - 24, T_s - 1] = \emptyset$, rows with missing future targets and boundary rows remain **completely disjoint**.

### Check 2: Do missing-target exclusions overlap with the 48-hour lag warmup?
- The 48-hour lag warmup set $W$ spans timestamps $t \in [0, 47]$ (`2013-03-01 00:00:00` to `2013-03-02 23:00:00`).
- For any prediction timestamp $t \in [0, 47]$ and horizon $h \le 24$, the required future target timestamp is:
  $$t + h \le 47 + 24 = 71 \quad (\text{i.e. within the first 72 hours of the dataset})$$
- An audit of the raw dataset reveals that **in all 12 monitoring stations, the first 72 consecutive hours (March 1–3, 2013) contain EXACTLY ZERO missing $PM_{2.5}$ measurements**.
- Therefore, for every timestamp in the lag warmup window, the future target is **always observed**.
- Consequently, **$M_h$ and $W$ never overlap**:
  $$\forall h \in \{1, 6, 24\}: \quad |M_h \cap W| = 0$$

### Check 3: Exact Number of Overlapping Rows
Across all 420,768 records and all 12 stations:
- **Pairwise Intersection $|B \cap M_h|$:** **0 rows** (for all $h \in \{1, 6, 24\}$)
- **Pairwise Intersection $|B \cap W|$:** **0 rows** (since $T_s - 24 = 35,040 \gg 48$)
- **Pairwise Intersection $|M_h \cap W|$:** **0 rows** (since first 72 hours are 100% complete)
- **Three-way Intersection $|B \cap M_h \cap W|$:** **0 rows**

**Exact Overlapping Rows:** **0 rows**. The three exclusion masks $B$, $M_h$, and $W$ are **mutually disjoint**.

### Check 4 & 5: Reconciliation of Usable Samples Across All 18 Combinations
Because the masks are mutually disjoint, the union size is strictly additive:
$$|B \cup M_h \cup W| = |B| + |M_h| + |W|$$

- **$h=1$ with lags:** $12 + 8,739 + 576 = \mathbf{9,327} \implies N_{\text{usable}} = 420,768 - 9,327 = \mathbf{411,441}$.
- **$h=1$ without lags:** $12 + 8,739 + 0 = \mathbf{8,751} \implies N_{\text{usable}} = 420,768 - 8,751 = \mathbf{412,017}$.
- **$h=6$ with lags:** $72 + 8,739 + 576 = \mathbf{9,387} \implies N_{\text{usable}} = 420,768 - 9,387 = \mathbf{411,381}$.
- **$h=6$ without lags:** $72 + 8,739 + 0 = \mathbf{8,811} \implies N_{\text{usable}} = 420,768 - 8,811 = \mathbf{411,957}$.
- **$h=24$ with lags:** $288 + 8,739 + 576 = \mathbf{9,603} \implies N_{\text{usable}} = 420,768 - 9,603 = \mathbf{411,165}$.
- **$h=24$ without lags:** $288 + 8,739 + 0 = \mathbf{9,027} \implies N_{\text{usable}} = 420,768 - 9,027 = \mathbf{411,741}$.

Every single number matches the exact integer output of `get_horizon_dataset()`.

### Check 6: Data Integrity Confirmation
Zero valid observations were modified, imputed, or artificially altered to produce this alignment. The results reflect the true, unmanipulated telemetry structure.

---

## 3. Station-Level Mask Breakdown & Reconciliation Table

The following per-station breakdown was computed from the actual data for each forecasting horizon:

### Station Mask Accounting for Horizon $h = 24$

| Monitoring Station | Raw Rows ($N$) | Boundary Mask $|B|$ | Missing Target Mask $|M_{24}|$ | Lag Warmup Mask $|W|$ | Intersections $|B \cap M|$, $|B \cap W|$, $|M \cap W|$ | Union Exclusions $|B \cup M \cup W|$ | Final Usable Samples (with lags) | Final Usable Samples (no lags) |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **Aotizhongxin** | 35,064 | 24 | 925 | 48 | 0 | 997 | **34,067** | **34,115** |
| **Changping** | 35,064 | 24 | 774 | 48 | 0 | 846 | **34,218** | **34,266** |
| **Dingling** | 35,064 | 24 | 779 | 48 | 0 | 851 | **34,213** | **34,261** |
| **Dongsi** | 35,064 | 24 | 750 | 48 | 0 | 822 | **34,242** | **34,290** |
| **Guanyuan** | 35,064 | 24 | 616 | 48 | 0 | 688 | **34,376** | **34,424** |
| **Gucheng** | 35,064 | 24 | 646 | 48 | 0 | 718 | **34,346** | **34,394** |
| **Huairou** | 35,064 | 24 | 953 | 48 | 0 | 1,025 | **34,039** | **34,087** |
| **Nongzhanguan**| 35,064 | 24 | 628 | 48 | 0 | 700 | **34,364** | **34,412** |
| **Shunyi** | 35,064 | 24 | 913 | 48 | 0 | 985 | **34,079** | **34,127** |
| **Tiantan** | 35,064 | 24 | 677 | 48 | 0 | 749 | **34,315** | **34,363** |
| **Wanliu** | 35,064 | 24 | 382 | 48 | 0 | 454 | **34,610** | **34,658** |
| **Wanshouxigong**| 35,064 | 24 | 696 | 48 | 0 | 768 | **34,296** | **34,344** |
| **TOTALS** | **420,768** | **288** | **8,739** | **576** | **0** | **9,603** | **411,165** | **411,741** |

---

## 4. Automated Test Verification

An automated test `test_exclusion_mask_accounting` was added to `tests/test_feature_engineering.py` to programmatically assert that:
$$\forall h \in \{1, 6, 24\}, \forall s \in \mathcal{S}: \quad |B \cap M_h| = 0, \quad |B \cap W| = 0, \quad |M_h \cap W| = 0$$
and that $N_{\text{usable}} = N_{\text{raw}} - (|B| + |M_h| + |W|)$ matches `len(X)`.

### Test Execution Output
```powershell
python -m pytest tests/
```

```
============================= test session starts =============================
platform win32 -- Python 3.13.5, pytest-8.3.4, pluggy-1.5.0
rootdir: C:\Users\Dell\Downloads\beijing+multi+site+air+quality+data
plugins: anyio-4.11.0
collected 20 items

tests\test_feature_engineering.py .............                          [ 65%]
tests\test_preprocessing.py .......                                      [100%]

======================== 20 passed, 1 warning in 2.49s ========================
```

All 20 test cases pass with zero errors.

---

## 5. Final Audit Verdict

# **DECISION: READY FOR PHASE 4**

The set-theoretic independence of the exclusion masks has been proven mathematically, verified empirically across all 12 stations, tested automatically via unit tests, and documented in full detail.

**Awaiting your approval to begin Phase 4: Baseline and Advanced Model Development.**
