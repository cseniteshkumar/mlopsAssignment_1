# Project Flowchart (Mermaid)

This file contains a Mermaid flowchart describing the data-to-deployment pipeline for this repository.

```mermaid
flowchart TD
  %% Data ingestion and reading
  A[Start]
  A --> B[Data: DataSet/heart_disease_cleveland.csv]
  B --> C[`src/dataRead.py`]
  C --> D{Data validation}
  D -->|passes| E[`src/dataClean.py`]
  D -->|fails| Z[Abort / Notify]

  E --> F[`src/dataEDA.py`]
  F --> G[Feature engineering & scaling]
  G --> H{Split: train / test}

  H --> I[`src/modelTrainPipeline.py`]
  I --> J[`src/modelTrain.py`]
  J --> K[Model training & evaluation]
  K --> L[Save models to `models/` & metrics to `outputs/modelTrain/`]
  L --> M[MLflow: `mlruns/` (experiments + artifacts)]

  M --> N{Select best model}
  N --> O[`src/modelDeploy.py`]
  O --> P[Save `deployed_model.joblib` and artifacts]

  P --> Q[`src/api.py`]
  Q --> R[Serve / REST endpoint]

  L --> S[`src/modelBatchPrediction.py`]
  S --> T[Batch predictions -> `batchProcessing/batchPrediction.csv`]
  T --> U[Outputs & reports: `outputs/eda/report.html`]

  %% Monitoring and retraining loop
  R --> V[Monitoring & metrics collection]
  V --> W{Trigger retrain?}
  W -->|yes| I
  W -->|no| End[End]

  Z --> End

  %% Notes
  classDef repoFiles fill:#f9f9f9,stroke:#333,stroke-width:1px;
  class C,E,F,I,J,O,Q,S repoFiles;

  click C "./src/dataRead.py" "Open file"
  click E "./src/dataClean.py" "Open file"
  click F "./src/dataEDA.py" "Open file"
  click I "./src/modelTrainPipeline.py" "Open file"
  click J "./src/modelTrain.py" "Open file"
  click O "./src/modelDeploy.py" "Open file"
  click Q "./src/api.py" "Open file"
  click S "./src/modelBatchPrediction.py" "Open file"
```

## How to view

- In VS Code: install the "Markdown Preview Mermaid Support" or "Mermaid Markdown Preview" extension and open this file to preview the diagram.
- On GitHub: GitHub supports Mermaid in Markdown (depending on repo settings); otherwise use a Mermaid renderer.
- To export to PNG/SVG: install `mmdc` (Mermaid CLI) and run:

```bash
# install (npm required)
npm install -g @mermaid-js/mermaid-cli
# export
mmdc -i docs/flowchart.md -o docs/flowchart.png
```

## Short summary

- Diagram maps the main files in `src/` to pipeline stages (data read, validation, cleaning, EDA, training, model registry, deployment, API, batch prediction).
- Includes a retraining loop triggered by monitoring.

## Next steps (optional)

- I can convert this to a PNG/SVG and add it to `docs/` if you'd like.
- I can refine the diagram with more detail (e.g., which functions inside each module, exact filenames for outputs, or adding k8s deployment flow using `k8s/`).
