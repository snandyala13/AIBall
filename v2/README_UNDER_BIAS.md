# NBA PROP UNDER BIAS ANALYSIS - QUICK START

## Files Generated

### 1. **under_analysis.py** (23 KB)
Main comprehensive analysis script. Run with:
```bash
python3 v2/under_analysis.py
```

**Outputs:**
- Under bias breakdown by prop type, line size, player role, season
- Month-by-month consistency analysis (25 months of data)
- Statistical significance testing (p = 2.60e-45)
- Distribution analysis: why the bias exists
- Season phase analysis: early/mid/late season patterns
- Cross-book vigorish analysis
- 7 different betting strategies with ROI calculations

**Key Findings:**
- 52.01% of outcomes are under (vs 50% expected)
- Bias is highly stable across all 25 months
- Threes show strongest under bias (+3.18%)
- P-value: 2.60e-45 (HIGHLY SIGNIFICANT)

### 2. **under_analysis_v2.py** (14 KB)
Enhanced strategy analysis focusing on profitability. Run with:
```bash
python3 v2/under_analysis_v2.py
```

**Outputs:**
- Breakeven math at -110 odds (requires 52.38% hit rate)
- Strategic filters with ROI calculations
- Model-based strategy: "DK projects under" (53.80% hit, +2.71% ROI)
- Combo strategies: Threes + Stars (53.65% hit, +2.43% ROI)
- Statistical validation of best filters
- Why 52% is not profitable at standard -110

**Key Insights:**
- Line > Projection filter: 53.80% hit rate (+2.71% ROI)
- Threes + Stars combo: 53.65% hit rate (+2.43% ROI)
- Best single filter: Threes (53.18% hit, +1.53% ROI)
- All-unders strategy: NOT PROFITABLE at -110

### 3. **UNDER_BIAS_REPORT.md** (16 KB)
Comprehensive written report with 12 sections:
1. Executive summary
2. Where under bias is strongest (breakdown by all dimensions)
3. Consistency analysis (month-by-month, season-by-season)
4. Root cause analysis (distribution, line placement, vig)
5. When bias is strongest (season phase, confidence)
6. Cross-book analysis (vigorish by prop type)
7. Strategy comparisons (7 different approaches)
8. Statistical significance testing
9. Critical odds analysis (why breakeven is 52.38%)
10. Root cause: why DK model is too high
11. Final recommendations (Tier 1-4 strategies)
12. Implementation roadmap + risk factors

## Key Findings (TL;DR)

### The Bias is Real
- 52.01% under vs 50% expected
- P-value: 2.60e-45 (essentially impossible by chance)
- 95% CI: 51.73% - 52.29%

### But Not All Unders are Profitable
- Breakeven at -110 requires 52.38%
- All unders: 52.01% hit = -0.70% ROI
- **NEED selective filters or better odds**

### Best Exploitation Strategies

**Tier 1 (Statistically Significant):**
- Line > Projection unders: 53.80% hit (+2.71% ROI)
- Threes + Stars unders: 53.65% hit (+2.43% ROI)

**Tier 2 (Medium Confidence):**
- All Threes unders: 53.18% hit (+1.53% ROI)
- High Lines (>20) unders: 52.93% hit (+1.05% ROI)

**Requires Line Shopping (<-110):**
- Any unders at -105: 52% hit becomes +1.52% ROI

### Implementation
1. **Immediate:** Threes + Stars filter (32 picks/day)
2. **Quick:** Line > Projection filter (158 picks/day)
3. **Extended:** Line shopping across all books
4. **Advanced:** Build proprietary model to beat DK

## Running the Analysis

```bash
# Full analysis (5-10 minutes)
cd /Users/suhaasnandyala/Documents/aiBall
python3 v2/under_analysis.py

# Enhanced strategy analysis (5-10 minutes)
python3 v2/under_analysis_v2.py

# View the report
cat v2/UNDER_BIAS_REPORT.md
```

## Data Source
- **File:** `/Users/suhaasnandyala/Documents/aiBall/outputs/backtest_all_enriched.json`
- **Size:** 54 MB
- **Records:** 122,224 legs
- **Period:** 2023-10 through 2026-06
- **Seasons:** 2023-24, 2024-25, 2025-26 + 2026 OOS

## Key Metrics by Strategy

| Strategy | Hit Rate | ROI | Volume | Confidence |
|----------|----------|-----|--------|------------|
| All Unders | 52.01% | -0.70% | 63,573 | Proven |
| Line > Projection | 53.80% | +2.71% | 58,038 | ⭐ BEST |
| Threes + Stars | 53.65% | +2.43% | 11,507 | ⭐ HIGH |
| All Threes | 53.18% | +1.53% | 22,657 | MEDIUM |
| High Lines | 52.93% | +1.05% | 23,000 | MEDIUM |

*Assumes -110 odds. All unders at -105 would be +1.52% ROI.*

## Statistical Validation

### Overall Bias
- **Sample:** 122,224 legs
- **Test:** Binomial test vs 50%
- **Result:** p = 2.60e-45 (HIGHLY SIGNIFICANT)
- **Power:** Essentially perfect (sample size huge)

### Best Filter (Threes + Stars)
- **Sample:** 11,507 legs  
- **Hit Rate:** 53.65%
- **Test:** Binomial vs 52.38% breakeven
- **Result:** p = 0.0032 (SIGNIFICANT)
- **Confidence:** 95%+ edge is real

## Next Steps

1. **Validate on OOS data** (2026 out-of-sample)
   - Does 53.80% persist in live 2026-2027 season?
   - Check for data leakage or overfitting

2. **Implement live system**
   - Connect to DK API
   - Auto-flag matching bets daily
   - Track actual results vs model

3. **Line shopping**
   - Get odds from all 19 books
   - Find -105 or better under odds
   - Estimate +0.5-1% improvement

4. **Model improvement**
   - Why is DK too high?
   - Recalibrate projections
   - Build better prediction model

## Contact & Questions

- **Analysis Location:** `/Users/suhaasnandyala/Documents/aiBall/v2/`
- **Report:** `UNDER_BIAS_REPORT.md`
- **Scripts:** `under_analysis.py`, `under_analysis_v2.py`
- **Data:** `/outputs/backtest_all_enriched.json`

