# Screenshots

The README references four captures. Drop them here with these exact names:

| File | What to capture |
|---|---|
| `chat.png` | A answered question showing citations, the grounding meter and the security panel |
| `dashboard.png` | The security dashboard after running the demo, so the counters are non-zero |
| `playground.png` | A blocked attack with its detectors, risk score and "why this decision" |
| `documents.png` | The document list with a quarantined chunk expanded in the inspector |

To generate interesting data first:

```bash
docker compose up -d
python scripts/seed_demo.py --url http://localhost:5173
```

Then sign in as the account the script prints and capture at ~1440px wide.
