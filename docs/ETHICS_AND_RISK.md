# Ethics and Risk Notes

## 1) Bias and Language Limitations
- The dataset is English-centric and may underperform on multilingual or code-mixed SMS.
- Regional slang, dialects, and emerging spam patterns can shift model behavior over time.
- Performance should be revalidated when deploying to new geographies or user populations.

## 2) Misclassification Risks
- False positives:
  - Legitimate messages may be flagged as spam.
  - Potential impact: user frustration, missed communication.
- False negatives:
  - Spam may pass through.
  - Potential impact: phishing exposure, fraud risk.

Mitigations:
- Tune threshold by business policy.
- Monitor error patterns regularly.
- Retrain periodically with fresh labeled data.

## 3) Privacy for Uploaded CSVs
- Batch CSV files are processed locally in the running app session.
- No cloud upload is required by the current project flow.
- Users should avoid including unnecessary personally identifiable information.
- Exported prediction files should be handled according to organizational data-retention policies.

## 4) Responsible Usage Guidance
- Use model output as decision support, not as the only final authority.
- Keep human override available in high-stakes workflows.
- Document threshold policy and review schedule in production.
