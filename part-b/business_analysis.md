# Part B — Business Case Analysis: Promotion Effectiveness at a Fashion Retail Chain

---

## B1. Problem Formulation

### B1(a) — ML Problem Formulation (3 marks)

**Target variable:** `items_sold` — the number of items sold at a given store in a given month under a given promotion.

**Candidate input features:**

| Feature category | Examples |
|---|---|
| Store attributes | store size, location type (urban/semi-urban/rural), monthly footfall |
| Promotion | promotion type (Flat Discount, BOGO, Free Gift, Category-Specific, Loyalty Points) |
| Context | month, season, local competition density |
| Customer demographics | average age, income band of catchment area |
| Historical performance | past items sold, past promotion response rates per store |
| Calendar | is_weekend, is_festival, is_month_end |

**Type of ML problem:** This is a **supervised regression problem**. The target variable (`items_sold`) is continuous and numerical, and we have labelled historical observations (past months with known outcomes). We frame it as regression rather than classification because the business wants to predict a quantity — how many items will be sold — not a category.

The downstream decision (which promotion to deploy) is made by selecting the promotion that maximises the predicted `items_sold`, treating the model as a **counterfactual simulator**: for each store × month × candidate promotion, predict the outcome, then pick the highest.

---

### B1(b) — Why Items Sold is a Better Target than Revenue (3 marks)

**Revenue = price × quantity sold.** Using revenue as the target conflates two distinct effects: how many items customers chose to buy (driven by the promotion's appeal), and the price at which those items were sold (which is directly manipulated by the promotion itself — a Flat Discount mechanically reduces revenue per unit regardless of customer behaviour).

If the model is trained on revenue, a Flat Discount will always appear to underperform relative to full-price promotions, not because it drives fewer purchases, but because it reduces the per-item revenue. The model would systematically penalise discounts in ways that do not reflect actual customer response.

`Items_sold` (sales volume) is free of this price distortion — it directly measures whether customers responded to the promotion by purchasing more. It is a cleaner signal of promotion effectiveness.

**Broader principle:** In real-world ML projects, the target variable should measure the **causal mechanism** the business actually wants to optimise (customer purchasing behaviour), not a downstream financial aggregation that mixes the intervention effect with the intervention's own mechanical impact on that aggregation. This is an instance of the general principle: *choose a target that is as close as possible to the behaviour you want to predict, and minimise confounding from the features themselves.*

---

### B1(c) — Modelling Strategy for Store Heterogeneity (2 marks)

A single global model assumes that all 50 stores respond to promotions in the same way — an assumption that is clearly violated when urban flagship stores and rural small-format stores have fundamentally different customer bases, footfall profiles, and competitive dynamics.

**Proposed strategy: Hierarchical / grouped modelling with store-level features**

Rather than one global model or 50 separate models (which would have insufficient data per store), the preferred approach is a **global model with rich store-level features** combined with **store-type segmentation**:

1. Cluster stores into groups (e.g., by location type, size, footfall band) and train a separate model per cluster. This gives each model enough data while allowing different response functions per store type.
2. Alternatively, use a **mixed-effects model** or a **tree-based model with store_id as a feature** — allowing the model to learn store-specific intercepts (baseline demand) while sharing information on promotion effects across stores.

The key justification is the **bias-variance trade-off**: a global model has low variance (many training examples) but high bias (wrong functional form for individual stores); a per-store model has low bias but very high variance (too few data points). The grouped or hierarchical approach finds the middle ground.

---

## B2. Data and EDA Strategy

### B2(a) — Joining Tables and Dataset Grain (4 marks)

**Table join strategy:**

The four tables are:
- `transactions` — one row per transaction (transaction_id, store_id, date, items_sold, promotion applied)
- `store_attributes` — one row per store (store_id, size, location_type, footfall, competition_density, demographics)
- `promotion_details` — one row per promotion type (promotion_id, promotion_type, discount_depth, mechanics)
- `calendar` — one row per date (date, is_weekend, is_festival, month, season)

**Join sequence:**
1. Aggregate `transactions` to **store × month × promotion_type** grain: sum `items_sold`, count transaction days.
2. Left-join `store_attributes` on `store_id`.
3. Left-join `promotion_details` on `promotion_type`.
4. Left-join `calendar` on `month` (or a representative date) to bring in seasonal flags.

**Final dataset grain:** One row = **one store × one month × one promotion type** (the promotion active in that store that month). This is the level at which recommendations will be made.

**Aggregations before modelling:**
- Sum of `items_sold` over all transactions in the store-month window.
- Average `competition_density` if it varies within a month (or use the month-start value).
- Count of weekend days and festival days in the month (from the calendar table) as additional numerical features.
- Compute a `historical_avg_items_sold` feature per store as a baseline demand signal.

---

### B2(b) — EDA Strategy (4 marks)

**Analysis 1: Items sold distribution by promotion type (box plots)**
- What to look for: Do certain promotions (e.g., BOGO) consistently produce higher or lower medians? Is variance high — indicating promotion effect is context-dependent?
- Influence on modelling: High variance within a promotion type signals the need for interaction features (promotion × location_type, promotion × season).

**Analysis 2: Heatmap of average items sold — store location × promotion type**
- What to look for: Do rural stores respond differently to Flat Discount vs urban stores? Are there promotion × location interactions?
- Influence on modelling: If the heatmap shows clear differential response patterns, interaction terms or segmented models are needed. It also identifies which promotion is the baseline winner by location type.

**Analysis 3: Time series of items sold by month (aggregated)**
- What to look for: Seasonality peaks (December, festival months), long-term trend, and whether the promotion mix used historically correlates with seasonal spikes (potential confounding).
- Influence on modelling: Strong seasonality → include month and season features. If promotions were preferentially run during high-demand periods, we need to control for this to avoid attributing season-driven sales to the promotion.

**Analysis 4: Correlation analysis — numerical features vs items_sold**
- What to look for: Which numerical features (footfall, competition_density, basket_size) have strongest linear correlation with items_sold? Any multicollinearity between store attributes?
- Influence on modelling: High multicollinearity → consider PCA or feature selection before training linear models. Weak correlations for some features → candidate for dropping to reduce noise.

**Analysis 5 (bonus): Promotion frequency analysis per store**
- What to look for: Were all promotions tested equally across all stores, or did certain store types predominantly use one promotion? Imbalanced promotion assignment → the model's promotion coefficients may be confounded with store type.
- Influence on modelling: If Loyalty Points were only ever run in urban stores, the model cannot disentangle urban store demand from Loyalty Points effectiveness. This would require caution in how promotion effect estimates are interpreted.

---

### B2(c) — Handling 80% No-Promotion Transactions (2 marks)

**Effect on the model:** If 80% of transaction rows have no promotion, a naively trained model will be heavily biased towards predicting the no-promotion baseline. When the model encounters promotion-specific features, it will have limited training signal to learn how different promotions differentially affect demand. It may learn to largely ignore promotion features and default to store-level and seasonal patterns, producing near-identical predictions for all five promotion types — making it useless for recommendation.

**Steps to address:**

1. **Reframe the modelling unit:** Aggregate to store × month × promotion grain (as described in B2a). At this grain, every row has an active promotion — the 80% imbalance in raw transactions disappears because no-promotion days are captured as a separate category or excluded from the recommendation task.
2. **Oversample promotion observations:** If the row-level imbalance persists after aggregation, apply SMOTE or simple oversampling on the minority promotion categories.
3. **Use promotion-stratified cross-validation:** Ensure each fold contains a representative mix of all five promotion types so the model is evaluated on its ability to differentiate between them.
4. **Separate baseline demand modelling:** Fit a no-promotion baseline model first, then model the **uplift** (additional items sold attributable to the promotion) as the target. This decomposition makes the promotion effect more learnable with limited promotion data.

---

## B3. Model Evaluation and Deployment

### B3(a) — Train-Test Split and Metrics (4 marks)

**Setting up the train-test split:**

With 3 years × 50 stores × monthly data, we have approximately 1,800 store-month observations. The correct approach is a **temporal holdout split**, not a random split.

- **Train set:** Months 1–30 (first 2.5 years, ~1,250 store-month rows)
- **Test set:** Months 31–36 (last 6 months, ~300 store-month rows)
- Optionally use **walk-forward cross-validation** (expanding window) across the training period to tune hyperparameters without touching the final test set.

**Why random split is inappropriate:** The same reasons as Q3 — future data would leak into training, temporal patterns would be disrupted, and the evaluation would not simulate actual deployment (where the model always predicts into the future). Additionally, stores have autocorrelated demand (this month's sales depend on last month's), so randomly shuffling would break that dependency structure.

**Evaluation metrics:**

| Metric | Formula | Business interpretation |
|---|---|---|
| **RMSE** | √(mean squared error) | Penalises large prediction errors; important if being wrong by 100 items is much worse than being wrong by 10 |
| **MAE** | mean absolute error | Average items-sold prediction error; directly communicable to marketing ("our predictions are off by X items on average") |
| **MAPE** | mean absolute % error | Scale-free; useful for comparing accuracy across stores of different sizes |
| **Rank accuracy** | Does model rank the winning promotion correctly? | For this recommendation use case, being directionally right (picking the best promotion) matters more than exact volume accuracy — a custom metric comparing whether the model's top-ranked promotion matches actual best promotion |

Rank accuracy is the most business-relevant metric here because the end output is a recommendation, not a precise sales forecast.

---

### B3(b) — Explaining Different Recommendations via Feature Importance (4 marks)

**Scenario:** The model recommends Loyalty Points Bonus for Store 12 in December and Flat Discount for Store 12 in March.

**How to investigate:**

1. **Extract feature importances** from the trained Random Forest (or use SHAP values for local explainability). Feature importances give a global view; SHAP gives per-prediction explanations.

2. **Generate SHAP force plots** for the two predictions (Store 12 in December, Store 12 in March). SHAP decomposes each prediction into the additive contribution of each feature, showing which features pushed the prediction up or down for each specific promotion.

3. **Compare the two predictions** by examining which features changed between December and March and how they affected the promotion rankings:
   - In December: `is_festival=1`, `month=12` (peak season), `competition_density` (possibly lower as competitors run their own holiday deals). The model has learned that during high-footfall festive periods, customers are already motivated to buy — a loyalty incentive converts occasional shoppers into repeat buyers more effectively than a one-time discount.
   - In March: `is_festival=0`, `month=3` (post-holiday dip), lower baseline footfall. A Flat Discount generates traffic that would not otherwise materialise, making it the better lever for a slow month.

**How to communicate to the marketing team:**

Present a simple table or bar chart showing the top 3 features that drove the recommendation difference. Frame it in business language: "In December, the model is telling us that festival-season footfall and the holiday purchasing mindset favour retention-oriented promotions. In March, footfall drops and price sensitivity rises — a direct discount is more likely to pull customers in." Avoid technical jargon; show the SHAP waterfall chart visually and annotate feature names in plain English.

---

### B3(c) — Deployment and Monitoring (4 marks)

**End-to-end deployment process:**

**Step 1 — Save the model**
Serialise the trained pipeline (including preprocessor + model) using `joblib.dump(pipeline, 'promotion_model_v1.pkl')`. Store the model artifact in a versioned model registry (e.g., MLflow, AWS S3 with versioning) alongside the training data snapshot, feature schema, and performance metrics at training time. This ensures reproducibility and rollback capability.

**Step 2 — Monthly data preparation**
At the start of each month, an automated pipeline:
- Pulls the previous month's store transaction data from the data warehouse.
- Aggregates to store × month grain.
- Joins with the latest store attributes, calendar flags, and promotion catalogue.
- Validates the schema and checks for missing values, out-of-range features, or new category levels (e.g., a new store opening, a new promotion type) — any schema mismatch triggers an alert before the model is called.
- Generates 5 input rows per store (one per promotion type) — the model scores all five and the highest predicted `items_sold` is surfaced as the recommendation.

**Step 3 — Generating recommendations**
`pipeline.predict(X_new)` is called on the prepared input. Output is a table of 50 stores × 5 promotions with predicted items_sold. The top-ranked promotion per store is written to the marketing dashboard.

**Step 4 — Monitoring and drift detection**
Three monitoring layers:

| Layer | What is monitored | Threshold / action |
|---|---|---|
| **Data drift** | Distribution of input features (e.g., `competition_density`, `footfall`) compared to training distribution using KS-test or PSI | Alert if PSI > 0.2 for any key feature; investigate cause |
| **Prediction drift** | Distribution of predicted items_sold over time | Alert if monthly mean prediction shifts > 2σ from historical average |
| **Model performance** | Actual items_sold vs predicted for each store-month once actuals are available (1-month lag) | Alert if rolling 3-month MAE exceeds 1.5× the training MAE |

**Retraining trigger:** If model performance degrades (MAE threshold breach), retrain on all available data up to the current month using the same pipeline and hyperparameters. If data drift is detected without performance degradation, log for review but do not immediately retrain — drift does not always cause performance drops. Schedule a full model review (including hyperparameter re-tuning) every 6 months regardless of alerts, to capture structural changes in customer behaviour or the competitive landscape.
