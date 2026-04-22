# Architecture Diagram

```mermaid
flowchart LR
    A[spam.csv] --> B[train_model.py]
    B --> C[Data Validation]
    C --> D[Feature Engineering + Model Training]
    D --> E[Quality Gates + Baselines]
    E --> F[artifacts/spam_model.joblib]
    E --> G[artifacts/metrics.json]
    G --> H[generate_model_report.py]
    H --> I[docs/model_report.md]
    F --> J[app.py]
    G --> J
    J --> K[Live Message Lab]
    J --> L[Batch CSV Studio]
    J --> M[Model Dashboard]
    J --> N[Session Log]
```

## Notes
- Training and inference share preprocessing from `text_preprocessing.py`.
- The app consumes both model artifact and metrics payload.
- Threshold sweep + keyword signals are surfaced in the dashboard for explainability.
