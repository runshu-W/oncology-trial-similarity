# Web Agent Interface Design

## Scope

Build a local web interface for the existing oncology trial similarity pipeline. The first version accepts one ClinicalTrials.gov-style trial JSON by upload or paste, runs the existing search and prior-borrowing rerank flow, and displays pipeline-native outputs in a readable browser UI.

This version does not fetch by NCT ID, rebuild indexes, edit the database, or make final clinical/statistical borrowing decisions.

## Recommended Approach

Use a small Python web app:

- `FastAPI` backend for local API routes and static file serving.
- Existing pipeline functions imported from `docs/oncology_trial_similarity_pipeline.py`.
- Plain HTML/CSS/JavaScript frontend with no React or Node build step.

This keeps the implementation close to the current Python pipeline and avoids a second application stack before it is needed.

## User Flow

1. User opens the local web app.
2. User uploads a `.json` file or pastes JSON text.
3. User clicks `Run Similarity Search`.
4. Backend validates that the input is parseable JSON.
5. Backend writes the query JSON to a temporary local file.
6. Backend calls the existing pipeline search with:
   - ClinicalBERT index by default.
   - `top_k = 10`.
   - `rerank_top_n = 100`.
7. Frontend displays:
   - Query title, phase, cancer summary, intervention summary.
   - Ranked historical trial candidates.
   - Pipeline score, retrieval score, suitability, suggested discount.
   - Rationale, red flags, dimension scores, and borrowable quantities.
8. User can download the raw JSON result and Markdown report.

## Backend Design

Routes:

- `GET /`: serve the web UI.
- `POST /api/search`: accept uploaded JSON file or pasted JSON string, run pipeline search, return result JSON plus rendered Markdown.
- `GET /api/health`: return index availability and configured default paths.

Configuration:

- Default index directory: `../artifacts/oncology_trial_similarity_clinicalbert` relative to the project root.
- Default embedding backend: read from the index metadata when possible.
- Default rerank: enabled.

Errors should be explicit and user-facing:

- Invalid JSON.
- Missing index files.
- Backend mismatch.
- ClinicalBERT dependency/model loading failure.

## Frontend Design

The page has three sections:

- Input panel: file upload, JSON textarea, run button, and status message.
- Result summary: query metadata and overall run settings.
- Candidate cards/table: top-ranked historical trials with expandable detail sections.

The UI should make red flags visually prominent and show that suggested borrowing discounts are sensitivity-analysis aids, not final prior weights.

## Testing

Minimum verification:

- Python syntax check for backend and pipeline.
- API health route works.
- Invalid JSON returns a clear error.
- A known local query JSON can run end-to-end against the existing ClinicalBERT index.
- Browser opens the local app and shows results after a successful run.

## Success Criteria

- A user can start one local command and open a browser page.
- A pasted or uploaded trial JSON produces the same pipeline-style output as the CLI search.
- Output includes suitability, discount, red flags, dimension scores, and borrowable quantities.
- No large artifacts are committed to the repo.
