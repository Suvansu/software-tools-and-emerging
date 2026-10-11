# MapleFreight: Late-Delivery Prediction and Slack Alerts

A notebook that predicts which in-transit shipments will arrive late and posts a ranked alert to a Slack channel, so dispatch can re-route or warn the customer before the delivery window closes.

 Python libraries used:  `os`, `warnings`, `numpy`, `pandas`, `matplotlib`, `seaborn`. Nothing else is imported. The train/test split, cross-validation, logistic regression, metrics and Slack call are all written by hand in small functions. Sending to Slack uses the `curl` command-line tool (see [Slack alert](#slack-alert)).

## The problem

MapleFreight moves about 8,000 shipments a week across Ontario, Quebec and the Prairies, under contracts with on-time guarantees. On-time performance fell from 94% to 88% this year, and the company only learns a shipment is late when the customer calls. By then service credits and re-delivery charges are already incurred.

 Goal:  flag shipments likely to be late while they are still in transit.

 Dataset:  `maplefreight_delivery_delay_dataset.csv`, 6,035 shipments and 23 columns. Target: `delivered_late` (1 = missed the promised window). About 21% are late, so "always predict on time" scores 79% accuracy and catches nothing.

## Repository layout

```
.
├── Assignment1.ipynb   main notebook
├── data/
│   └── maplefreight_delivery_delay_dataset.csv   the dataset
├── docs/
│   └── slack_alert.png                           screenshot of the delivered alert
├── .env.example                                  template for the webhook (committed)
├── .gitignore
└── README.md
```

`.env` holds the real webhook URL and is  never committed .

## How to run

1. Install the libraries: `pip install numpy pandas matplotlib seaborn jupyter`
2. Put the CSV in `data/`.
3. Start Jupyter  from this folder  and open the notebook. The notebook looks for `.env` in the folder it is running from.
4. Choose  Kernel > Restart & Run All .

Without a `.env` file the notebook still runs to the end. The Slack step prints a dry run and sends nothing. The notebook was developed with pandas 3.x.

## Notebook map

 Section  What it does 
------
 1-3. Context, problem, objectives  Business background and what the model must do 
 4. Data understanding  Shape, types, summary statistics, late rate 
 5. EDA  Data-quality audit, late rate by carrier, weather, route and pickup, leakage check 
 6. Preprocessing  Duplicates, spelling, unit errors; 6.1 outlier check; 6.2 outlier capping 
 7. Feature engineering  Three dispatch-time features 
 8. Data preparation  Column choice, stratified 80/20 split, imputation, capping, scaling, encoding 
 9. Data modelling  Hand-written logistic regression, baseline vs balanced, 5-fold cross-validation 
 10. Model evaluation  Precision, recall, F1, confusion matrices, PR curve, alert threshold 
 11. Slack alert and explainability  Slack steps (first cells), then 11.1 coefficients and 11.2 recommendations 

## What the data needed

 Problem  Count  Fix 

 Duplicate rows  32 exact copies, plus 3 repeated shipment IDs with conflicting values  Dropped; first record kept for conflicting IDs (6,035 to 6,000 rows) 
 Inconsistent spelling  `service_level` mixed `Standard` and `standard`  Converted to Title Case 
 Unit error  40 weights between 170,000 and 3,000,000 kg, about 1,000 times too large (grams entered as kg)  Divided by 1,000 
 Missing values  `driver_experience_years` 210, `weather_condition` 150, `traffic_index` 121, `fuel_cost_cad` 240  Median or "Unknown", learned from training data only 
 Outliers  343 of 6,000 rows (5.7%) have a value beyond 3 standard deviations  Capped at mean +/- 3 SD, learned from training data only 
  Data leakage   `actual_transit_hours` is recorded on delivery and correlates about 0.49 with the target   Excluded from every model  

Also left out of the model: `shipment_id`, and the two city columns (distance, stops and transit time already describe the route).

## Method

-  Split:  stratified 80/20 (4,800 train, 1,200 test), same 20.6% late rate in both. The test set is used once, for the final scores.
-  Preprocessing inside the pipeline:  every fill value, cap and scaling value is learned from the training rows only, so nothing leaks from the test set.
-  Model:  logistic regression written by hand (Newton's method with an L2 penalty).
-  Imbalance:  `class_weight="balanced"`, so each late shipment counts about 2.4 times as much as an on-time one. No resampling, so no synthetic rows end up in validation folds.
-  Comparison:  5-fold cross-validation on the training data, scored by PR-AUC, which stays honest when classes are imbalanced.
-  Features added:  `is_winter`, `pickup_over_30`, `stops_per_hour`.

## Results

Cross-validated PR-AUC on the training data:

 Model  CV PR-AUC 
------
 Baseline: plain logistic regression  0.575 
 Balanced logistic regression ( chosen )  0.575 
 Balanced logistic regression + new features  0.574 

All three tie within noise (about 0.02 between folds). Class weights do not change how shipments are ranked, so PR-AUC cannot separate the baseline from the balanced model. The difference shows up in recall below. The new features added nothing, so the simplest model was kept.

Test set (1,200 shipments, 247 late):

 Model  Threshold  Precision  Recall  F1  PR-AUC  ROC-AUC 
---------------------
 Always "on time"  0.50  0.000  0.000  0.000  0.204  0.500 
  Baseline  (plain logistic regression)  0.50  0.699  0.320  0.439  0.583  0.824 
  Final  (balanced)  0.50  0.428  0.733  0.540  0.585  0.824 
  Final  (balanced), chosen alert threshold   0.63    0.517    0.563    0.539   0.585  0.824 

Confusion matrices on the test set:

  Predicted on time  Predicted late 
---------
  Baseline, actually on time   919  34 
  Baseline, actually late   168  79 

  Predicted on time  Predicted late 
---------
  Final at 0.63, actually on time   823  130 
  Final at 0.63, actually late   108  139 

The baseline has the better accuracy (0.832) but catches only 79 of the 247 late shipments. The final model at 0.63 catches 139 of 247 with 130 false alarms. The ROC-AUC of 0.82 means the model ranks risky shipments well, but it cannot be both very precise and very sensitive. It supports dispatch decisions rather than replacing them.

## Alert threshold: 0.63

The threshold is the risk score above which dispatch is alerted. It is chosen on out-of-fold training predictions, so the test set stays a clean final check.

-  Cost-only optimum:  0.49, which would alert on 36% of shipments (precision 0.43). Too noisy; dispatch would learn to ignore the channel.
-  Rule added:  at least half of alerts must be real late shipments (precision of 0.50 or more).
-  Chosen 0.63:  precision 0.50, recall 0.59 and about 24% of shipments flagged (roughly 1,900 a week at 8,000 shipments). On the test set, 269 of 1,200 shipments (22%) are flagged.

 The dollar figures are assumptions, not MapleFreight data:  a missed late shipment costs CAD 300, an alert costs CAD 30 of dispatch time, and dispatch saves 50% of flagged late shipments. On these assumptions the expected cost per shipment is about CAD 50.71 against CAD 61.69 with no system (about 18% lower). That saving is only as good as the assumptions. They sit in one cell (`COST_MISSED_LATE`, `COST_ALERT`, `SAVE_RATE`) so finance can replace them.

## Main drivers of late delivery

Late rates from the cleaned data, confirmed by the model coefficients:

 Driver  Finding 
------
 Carrier  Contract owner-operators 36% late, MapleFreight fleet 13%. The gap holds in every weather condition. 
 Weather  Clear 14%, snow 33%, storm 46%. Storm is the largest single coefficient. 
 Pickup  Picked up more than 30 minutes late: about 27% late, against about 15% for early or on-time pickups. 
 Stops  3 to 5 stops: 26% late, against 12% with none. 
 Service level  Economy 27% late, Express 15%. 
 Route  Longer distance, heavier traffic and longer scheduled transit all raise risk. 
 Customer tier  No difference between Gold, Silver and Bronze. 

Coefficients show direction and relative size, not cause and effect.

## Slack alert

For the demo, the held-out test shipments stand in for the shipments currently in transit. In production the same code would score the live in-transit batch.

![Slack alert](docs/slack_alert.png)

Example message (numbers vary slightly between runs):

```
:rotating_light: *269 of 1200 in-transit shipments are at high risk of being late*
Top 5 by risk score:
1. SHP-105192  Ottawa ON  Contract Owner-Op  risk 99/100
2. SHP-100936  Calgary AB  PrairieLine  risk 99/100
3. SHP-100079  Regina SK  Contract Owner-Op  risk 98/100
4. SHP-103230  Regina SK  Contract Owner-Op  risk 98/100
5. SHP-102910  Edmonton AB  QuebecExpress  risk 98/100
Re-route or warn the customer before the delivery window closes.
```

The message leads with the count, then gives ID, destination, carrier and a 0 to 100 risk score for the top five, so it reads on a phone. The score ranks shipments. It is not an exact probability, because the model is trained with class weights.

### One-time setup

1. Create a free Slack workspace (or use one you have) and a channel named `#dispatch-alerts`.
2. Go to `api.slack.com/apps`, choose  Create New App > From scratch , name it and pick the workspace.
3. Open  Incoming Webhooks  and switch  Activate Incoming Webhooks  on.
4. Click  Add New Webhook to Workspace , choose `#dispatch-alerts`, click  Allow , and copy the URL.
5. Create a file named `.env` next to the notebook with one line (your own URL, no quotes):
   ```
   SLACK_WEBHOOK_URL=<your webhook URL>
   ```
6. Add `.env` to `.gitignore`. Commit `.env.example` with a placeholder value instead.
7. Restart the kernel after creating `.env`, because the notebook reads it once at the start of the Slack step.

### How the notebook sends it

 Step  What happens 
------
 1  Reads `.env` line by line into `os.environ` and prints only `True` or `False`, never the URL 
 2  Keeps shipments scoring at or above the threshold, sorted riskiest first 
 3  Builds the message text 
 4  Wraps it as JSON, `{"text": "..."}`, using string replacements (the `json` library is not imported) 
 5  Posts it with `curl` via `os.popen`. The shell fills in `$SLACK_WEBHOOK_URL`, so the URL never appears in the notebook or its output 
 6  Deletes the temporary `slack_payload.json` 

`curl` ships with macOS, Linux and Windows 10 or newer. The Windows path (`%SLACK_WEBHOOK_URL%`) was not tested.

### If it does not send

 Slack's reply  Meaning  Fix 
---------
 `ok`  Delivered  Nothing to fix 
 `invalid_payload`  The JSON is broken  Re-run steps 3 and 4 and read the printed payload 
 `no_service` or `invalid_token`  Wrong or revoked webhook URL  Copy it again from the Slack app page and update `.env` 
 `channel_not_found` or `channel_is_archived`  The channel was removed  Add a webhook for an active channel 
 No reply  `curl` could not reach Slack  Check the internet connection and that `curl` is installed 

## Keeping the webhook secret

- The webhook URL is a credential: anyone who has it can post to your channel.
- It lives only in `.env`, which is git-ignored. `.env.example` holds a placeholder, never a real URL.
- Add `slack_payload.json` to `.gitignore` too, in case a run stops before the cleanup step.
- Before committing, confirm that `git status` does not list `.env`, and run `git grep -n "hooks.slack" -- . ":!README.md"`. It should find nothing.
- If a webhook is ever exposed, revoke it on the Slack app page and create a new one.

## Recommendations

1. Fix the carrier gap first.  Review owner-operator contract terms and onboarding, and avoid them for Express freight where possible.
2.  Plan for weather.  When the forecast is snow or storm, add time to the promised window, use the carriers with the best winter record, and warn customers at booking.
3.  Treat a late pickup as an early warning.  Check shipments picked up more than 30 minutes late at pickup time, while there is still time to re-plan the route.
4.  Quote realistic windows  for multi-stop, long-distance and Economy shipments. Added slack costs less than paying credits.
5.  Use the alert as a ranked watch-list.  About a quarter of shipments are flagged, so dispatch should work from the highest score down and go further only as capacity allows. Re-tune the threshold once finance confirms real costs.
6.  Improve the data.  Confirm what `is_cross_border` means (every flagged shipment has a Canadian destination, and 466 shipments have the same origin and destination city). Add a pickup date so the model can be tested over time, and a live ETA or GPS feed. Better live signals are likely to help more than a more complex model.

## Limitations

- One random split of 6,000 shipments. There is no date column, so the model could not be tested forward in time.
- The cost figures and the 50% save rate are assumptions.
- The risk score is a ranking, not a calibrated probability.
- The Slack demo scores held-out test shipments, not a live in-transit feed.
- Cross-validated scores differ by about 0.02 between folds, so small differences between models are noise.

## AI use

 Assistant used:  Claude Help me with code, written findings and this README. 

 One case where the output was wrong:  an early draft of the feature-engineering finding said `pickup_delay_share` and `required_speed_kmh` were the most informative engineered features. Running the code showed their correlation with the target was close to zero (0.03 and -0.04), and the model comparison showed engineered features did not improve cross-validated PR-AUC. The claim was rewritten to match the evidence, and those two features were dropped. A similar assumption, that a gradient-boosted model would beat logistic regression, also failed in cross-validation and was reported as such. 
