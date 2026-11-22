# CareLoop Clinics — No-Show Guard

## Project overview
A reproducible clinic appointment analytics project covering Python data cleaning, SQL investigation, no-show risk modelling, and structured follow-up-note analysis.

## Source data
The original, unmodified files supplied with the assessment are included in `data/`:
- `appointments.csv` — 1,020 source rows
- `clinics.csv` — 8 clinics
- `followup_notes.csv` — 30 notes (the first 10 are used for Part 4)

## Reproducing the project
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```
Run notebooks in order:
1. `01_ingest_clean.ipynb`
2. `02_queries.ipynb`
3. `03_model.ipynb`
4. `04_genai.ipynb`

Part 1 and Part 2 outputs can be regenerated without API credentials. Part 3 trains and evaluates the candidate models. For the genuine Gemini API step in Part 4, set `GEMINI_API_KEY` in your environment before running the Part 4 notebook:
```powershell
$env:GEMINI_API_KEY="YOUR_KEY"   # Windows PowerShell
```
```bash
export GEMINI_API_KEY="YOUR_KEY" # macOS/Linux
```
Never put the key in a notebook cell or commit it.

## Observed findings from the supplied data

### Part 1 — Data preparation
- 1,020 appointment rows were read; no numeric parse failures were found.
- The cleaning pipeline retained 994 rows after removing duplicate records and the row with a missing appointment date.
- 26 distance values were capped using Tukey fences; age, distance and fee missing values are handled using the documented median/imputation rules.
- Clinic master data was joined to appointments and lead-time, age-band and distance-band features were generated.

### Part 2 — SQL analysis
- Overall no-show rate in the clean table: **21.93%** (218 of 994 rows; 4 rows have missing status and are counted in the denominator).
- No-show counts were highest at **C02 (39)** and **C04 (38)**.
- The 31+ day lead-time band had a **59.1%** no-show rate, compared with **16.1%** for 0–7 days and **23.1%** for 8–30 days.
- Appointments with `reminder_sent = No` had a **30.7%** no-show rate versus **15.2%** when `reminder_sent = Yes`.

These are descriptive associations, not evidence that reminders or lead time alone cause attendance outcomes.

### Part 3 — Risk model
The first 70% of eligible rows in source order were used for training and the final 30% for the held-out test. Direct identifiers and target fields were excluded from model inputs. Candidate models were logistic regression, a depth-3 decision tree and random forest. The model and threshold are selected using training-only cross-validation / out-of-fold probabilities.

- Selected model: **logistic regression**
- Selected threshold: **0.20**
- Held-out test ROC-AUC: **0.736**
- Held-out test recall: **0.896**
- Held-out test precision: **0.291**
- Held-out test accuracy: **0.485**

The threshold favours catching more potential no-shows, but the low precision means many flagged appointments will still attend. Use the score for supportive outreach, not punitive decisions.

### Part 4 — Follow-up notes
The first 10 notes include 9 nonblank notes and one blank note (N004). N004 is marked `skipped_blank` and does not trigger an API call.

**Important execution status:** The included `outputs/notes_analysed.csv` contains transparent offline, rule-based preliminary classifications so the data structure and appointment merge can be inspected without an API key. The `analysis_method` column marks these rows as `offline_rule_based`. They are **not Gemini-generated responses** and must not be represented as such. To complete the assessment's GenAI requirement, run `04_genai.ipynb` with a valid `GEMINI_API_KEY`; it calls Gemini, validates the JSON with Pydantic, retains each raw response, and marks invalid replies without repairing them. The notebook also includes deliberately invalid validation tests and a V1/V2 prompt comparison for N007 and N009.

## Business recommendation
Prioritise low-friction reminder confirmation and follow-up for appointments booked more than 30 days ahead: this band had a 59.1% no-show rate versus 16.1% for 0–7 day lead times. Review reminder delivery because the no-reminder group had a 30.7% no-show rate compared with 15.2% in the reminder group. Start a small, monitored outreach pilot at C02 and C04, which recorded 39 and 38 no-shows respectively. The logistic-regression model achieved 0.736 test ROC-AUC and 0.896 recall at a 0.20 threshold, but its 0.291 precision means staff should use it to prioritise a friendly confirmation call rather than assume a patient will miss the visit. The first-ten note sample also suggests multiple operational causes—reminder gaps, transport, fees, illness and work schedules—so interventions should be tailored and evaluated rather than applying one blanket policy.

## Limitations
1. The appointment data are observational; the reminder/no-show relationship is not a causal estimate, and unmeasured factors may explain the differences.
2. The final 30% source-order holdout may reflect ordering effects and is a single evaluation sample; performance may change over time or across clinics.
3. Part 4 has only ten sampled notes, one of which is blank. The currently included classifications are rule-based placeholders until the required Gemini API run is completed; even after that, model-generated note labels require human review.

## Repository contents
- `data/` — original input CSVs
- `src/pipeline.py` — reusable cleaning, SQL, modelling and GenAI functions
- `sql/02_queries.sql` — SQL analyses
- `01_ingest_clean.ipynb` through `04_genai.ipynb` — assessment notebooks
- `outputs/` — generated analysis files, metrics, database and merged table
