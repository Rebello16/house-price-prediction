## Where the model fails, and why

Final model: tuned XGBoost. Held-out test set of 292 houses: RMSLE 0.114, median absolute error 6.3%,
75.7% of predictions within 10% of the true price. Those are good averages, but the errors are not spread evenly.

**1. The cheapest houses (under $100k).** They are 7% of the test set but cause **32% of the total squared log error**.
Mean absolute error is 20.8% (vs 6.7-8.6% in every other price band) and the model *over*-predicts them by 18% on average.
Only 93 of the 1,166 training houses sell for under $100k, and what separates a $40k house from a $75k house
(interior state, deferred maintenance, distressed sale) is mostly not in the columns. The model regresses
toward the average. Worst case: a 1920 `IDOTRR` house, 1,317 sq ft, sold for $40,000, predicted $76,105 (+90%).

**2. Low-quality and old houses.** Houses rated `OverallQual <= 4` are 10% of the test set but 41% of the error
(RMSLE 0.233 vs 0.114 overall). Houses built before 1950 are 23% of the set and 48% of the error (RMSLE 0.165).
Two houses with the same rating and age can be a fixer-upper or a lovingly renovated home, and `OverallQual`
and `OverallCond` are coarse 1-10 human ratings with no photos or renovation details behind them.

**3. Unusual sales.** Non-`Normal` sale conditions (abnormal sale, family sale, allocation...) are 21% of houses but 28% of the
error; abnormal sales have RMSLE 0.160 vs 0.109 for normal ones. In a foreclosure or family transfer the price reflects the
*circumstances*, which no property feature can reveal.

**4. The expensive tail (over $300k).** Only mild: about 4% under-prediction on average across 23 houses. Tree models cannot
extrapolate beyond what they saw, so rare, very large or very luxurious houses are pulled down toward the bulk of the data.

**5. The metric is dominated by a few houses.** The 10 worst predictions (3.4% of the test set) account for **39%** of the squared
log error. Without them the RMSLE would be 0.091 rather than 0.114. This is why one lucky or unlucky test sample moves the score,
and why the cross-validated figure (0.112 +/- 0.012 across folds) is as important as the single test number.

**Sensitivity to the outlier decision.** Two partial-sale mega-houses (Ids 524 and 1299: 4,676 and 5,642 sq ft sold for $184,750
and $160,000) were removed before the split, as the dataset's author recommends. The model predicts roughly $706k and $781k for them,
which is what their size implies. If they were left in the test set the RMSLE would be **0.166** instead of 0.114. The headline
number is therefore valid for "ordinary, fully-built houses", not for sales where the price is disconnected from the property.

**The 80% prediction band is uniform.** It is +/-11.5% for every house and covers 78.8% of the test houses overall, but the real
uncertainty is much larger for cheap, old and unusual sales than for a typical mid-range house.

### What would improve it
- More low-price examples, or features about condition and renovation (photos, permit history).
- A robust loss (Huber / quantile) or a separate model for the low end.
- Segment-aware (conformal) prediction intervals instead of one uniform band.
- Neighborhood-level price trends, since this data covers only 2006-2010 in one city.
