# Deployment

The frontend and the backend go to different places, and the reason is not
preference.

## Why the backend is not on Vercel

Vercel runs Python as serverless functions. Four things this application does
are incompatible with that model, and none of them are worked around by
configuration:

| What the backend does | On serverless |
|---|---|
| Persists uploaded documents under `STORAGE_DIR` | `/tmp` only, discarded between invocations — upload succeeds, retrieval later finds nothing |
| Holds a SQLAlchemy connection pool | One pool *per function instance*; a burst exhausts the database's connection limit |
| Rate-limits in process | Each instance keeps its own counter, so the limit is multiplied by the instance count — the [documented per-process limitation](security.md#rate-limiting) becomes total |
| Runs `alembic upgrade head` at start | No startup hook; migrations would need a separate mechanism |

A single request also spans retrieval, generation and five guardrail stages,
which sits badly against a function duration cap. And `GROUNDING_METHOD=hybrid`
pulls ~2 GB of torch, far past the bundle limit.

The frontend has none of these properties. It is a Vite build — static files —
which is precisely what Vercel is good at.

So: **frontend on Vercel, backend and Postgres on Render.**

---

## What the demo does without an API key

More than you would expect, and this is worth understanding before paying for
anything.

The security layers are deterministic server-side code — pattern matching,
noisy-OR arithmetic, SQL predicates, checksum validation. [Four measured
configurations](evaluation.md#results) show detection rate, false-positive
rate, indirect-injection detection and quarantine precision **identical**
across the offline stub and a real model. So on the default `echo` provider:

| Works fully | Degraded |
|---|---|
| Security Playground (23 live attacks) | Chat answers are extractive, not generated |
| Security Dashboard and event log | Retrieval matches words, not meaning |
| Injection detection, PII redaction, authorisation | Answer quality metrics are a floor |
| Document ingest and quarantine | |

A deployment on `echo` is an honest demo of the thing this project is actually
about. Add a key later if you want generated prose — any OpenAI-compatible
endpoint works (`LLM_PROVIDER=openai` plus `LLM_BASE_URL`), including free
tiers such as Groq.

---

## 1. Backend and database (Render)

```
Render dashboard -> New -> Blueprint -> select this repository
```

[`render.yaml`](../render.yaml) provisions the web service, a 1 GB persistent
disk for uploads, and a PostgreSQL 16 instance, and generates `JWT_SECRET_KEY`
itself so no secret lands in git.

**Check pgvector before relying on it.** Migration `0001` runs
`CREATE EXTENSION IF NOT EXISTS vector` and the schema uses a native
`vector(n)` column — the deploy fails at migration time if the extension is not
available. Render's managed Postgres supports it, but verify on your plan
before assuming. If it is not available, point `DATABASE_URL` at
[Neon](https://neon.tech) or [Supabase](https://supabase.com) instead; both
support pgvector on their free tiers, and Neon can be provisioned from the
Vercel marketplace.

`DATABASE_URL` is normalised at startup, so the `postgres://` form these hosts
hand out is accepted and rewritten to `postgresql+psycopg://`.

Leave `CORS_ORIGINS` unset for now — step 3.

---

## 2. Frontend (Vercel)

```
Vercel -> Add New -> Project -> import this repository
```

| Setting | Value |
|---|---|
| Root Directory | `frontend` |
| Framework | Vite (detected) |
| `VITE_API_BASE_URL` | `https://<your-render-service>.onrender.com/api/v1` |

[`frontend/vercel.json`](../frontend/vercel.json) handles the SPA rewrite —
without it, a refresh on `/playground` returns 404 because no such file exists
— and sets the same hardening headers the API sends.

> **No Content-Security-Policy is set on the frontend.** A meaningful one needs
> `connect-src` to name the API origin, which is per-deployment and cannot be
> expressed in a static header block. A wildcard CSP would be decoration rather
> than a control, so there is none rather than a misleading one. The API sets
> its own CSP; see [security.md](security.md).

---

## 3. Connect the two

The two URLs depend on each other, so this is deliberately last:

1. Copy the Vercel deployment URL.
2. Set `CORS_ORIGINS` on the Render service to exactly that origin —
   `https://your-app.vercel.app`, no trailing slash, no path.
3. Redeploy the backend.

Vercel gives every preview branch its own URL, and none of them will be in
`CORS_ORIGINS`. Previews will fail their API calls unless you add them.

---

## 4. Claim the admin account immediately

**Do this the moment the backend is live.**

The first account ever registered becomes admin; every later one is a normal
user. On a public deployment with `ALLOW_REGISTRATION=true`, that means **the
first stranger to find your URL becomes the administrator** — with access to
the Security Dashboard, the event log and the evaluation endpoint.

Register your own account first. Then decide:

- **Portfolio demo, open to visitors** — leave registration on so a recruiter
  can try it. Admin is already yours.
- **Locked down** — set `ALLOW_REGISTRATION=false` and redeploy.

---

## Free-tier behaviour worth knowing

- **Render free services sleep after inactivity.** The first request after
  idling takes ~50 s while the container starts. A recruiter clicking a cold
  link will see a hang. Consider a paid instance, or say so on the page.
- **Render free Postgres expires.** Check the current retention policy; the
  database is deleted when it lapses, taking the demo with it.
- **The 1 GB disk is not backed up.** It holds demo uploads, which are
  reproducible with `scripts/seed_demo.py`.

## Verifying a deployment

```bash
curl https://<service>.onrender.com/health/ready
```

`/health/ready` checks the database, the migration revision, and that
`EMBEDDING_DIMENSIONS` matches the migrated `vector(n)` column — the dimension
mismatch that was [previously an opaque 500](evaluation.md#finding-5--three-bugs-found-only-by-running-on-postgresql).

Then walk the whole scenario end to end against the live API:

```bash
python scripts/seed_demo.py --url https://<service>.onrender.com
```

It creates two users, uploads a clean corpus and a deliberately poisoned
document, and prints what each guardrail did. Nothing in it is stubbed.
