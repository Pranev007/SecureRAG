# Deployment

The frontend and the backend go to different places, and the reason is not
preference.

## Why the backend is not on Vercel

Vercel runs Python as serverless functions, and four things about this
application sit badly against that model:

| What the backend does | On serverless |
|---|---|
| Holds a SQLAlchemy connection pool | One pool *per function instance*; a burst exhausts the database's connection limit unless a pooler sits in front |
| Rate-limits in process | Each instance keeps its own counter, so the effective limit multiplies by the instance count — the [documented per-process limitation](security.md#rate-limiting) becomes total |
| Runs `alembic upgrade head` at start | No startup hook; migrations need a separate mechanism |
| Would run `GROUNDING_METHOD=hybrid` | ~2 GB of torch, far past the bundle limit |

A request spanning retrieval, generation and five guardrail stages also sits
badly against a function duration cap once a real LLM is configured.

**None of these are individually fatal**, and it is worth being straight about
that: with a connection pooler, migrations run out-of-band and the `echo`
provider, a Vercel deployment of this backend would probably work. What it
would not do is behave like the system the rest of this repository describes —
rate limiting in particular stops being a control at all. The recommendation
here is a judgement about fidelity, not a hard incompatibility.

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
about — and `LLM_PROVIDER=echo` is the one-variable way back to it at any time.

### Adding a real model

[`render.yaml`](../render.yaml) points at **Groq** by default: its free tier is
enough for a public demo, and being hosted it is faster than local inference
rather than slower. One provider class covers OpenAI, Groq, Together,
OpenRouter, vLLM and LM Studio, so changing vendor is a base URL and a model
name.

The key is the only part not in the blueprint. Set it in the Render dashboard:

```
LLM_API_KEY = <your key>
```

`sync: false` on that variable means the blueprint neither carries the value
nor overwrites what you set, so it stays out of git permanently.

> **Without a key the chat returns a clean `502` — "The language model service
> is currently unavailable."** Not a crash, and no internal detail leaks. The
> Playground, Dashboard, ingestion and every guardrail keep working, because
> none of them call a model.

**Set the key before the blueprint syncs** and there is no gap at all: the
variable sits unused while the provider is still `echo`, then takes effect the
moment it flips.

Two things not to change casually:

- **`LLM_MODEL`** — Groq retires model names, and a stale one is a 404 on every
  request. Check [the model list](https://console.groq.com/docs/models) first.
- **`EMBEDDING_PROVIDER`** — the `vector(n)` column is sized from
  `EMBEDDING_DIMENSIONS` at migration time, so switching embedders needs a
  fresh database and a re-upload of every document. `/health/ready` reports the
  mismatch if the two disagree. Swapping the *LLM* has no such constraint.

---

## Choosing a host

Because the service is stateless apart from Postgres, the only hard
requirements are: **runs a container**, **long-lived process** (for the
connection pool and the startup migration), and **a Postgres with pgvector**.
That is a low bar, and it means compute and database can be chosen separately.

Free tiers move constantly and several shrank during 2025-26 — Fly.io ended its
free allowance for new accounts, Railway became a $5 trial credit, and Koyeb's
free compute status is reported inconsistently. **Verify current terms before
committing**; the notes below were accurate as of August 2026.

### Compute

| | Free? | Cold start | Notes |
|---|---|---|---|
| **Render** | Yes, indefinitely | **~50 s** | Most accessible; no card. The sleep is the one real drawback |
| **Google Cloud Run** | Generous free tier | ~1-5 s | Card required. Scales to zero but wakes fast — the best answer to Render's sleep |
| **Oracle Cloud Free Tier** | Yes, "always free" | None (always on) | 4 ARM cores / 24 GB RAM — far more machine than this needs. You administer the VM yourself, and signup can be awkward |
| **Fly.io** | No longer, for new accounts | ~1-3 s | ~$2/mo shared-CPU. Worth it if the cold start matters |
| **Railway** | $5 trial credit | Low | Best DX; not really free past the credit |

### Database — this is the choice that matters more

For a portfolio demo the failure mode is not slowness, it is **the database
disappearing between the day you deploy and the day someone opens the link**.

| | Free storage | pgvector | Idle behaviour |
|---|---|---|---|
| **Neon** | 0.5 GB/project | Yes | Scales to zero, **resumes instantly** |
| **Supabase** | 500 MB | Yes | **Pauses after ~1 week idle** — needs manual resume |
| **Render Postgres** | Yes | Yes | Free instances expire; check the current retention policy |

**Recommendation: Render for compute, Neon for Postgres.** Render is the least
friction for the container, and Neon removes the expiry risk that makes a
free-tier demo quietly die. Neon can also be provisioned from the Vercel
marketplace, so the whole stack stays across two dashboards.

To use Neon instead of Render's database, delete the `databases:` block from
[`render.yaml`](../render.yaml) and set `DATABASE_URL` to the Neon connection
string — the `postgres://` form is normalised at startup, and the **pooled**
endpoint is the right one given each instance holds a SQLAlchemy pool.

If the ~50 s wake is unacceptable for a link on a CV, **Cloud Run** is the
upgrade: same container, same `Dockerfile`, cold start in seconds. It needs a
card even when the usage is free.

---

## 1. Backend and database (Render)

```
Render dashboard -> New -> Blueprint -> select this repository
```

[`render.yaml`](../render.yaml) provisions the web service and a PostgreSQL 16
instance, and generates `JWT_SECRET_KEY` itself so no secret lands in git.

**No volume is needed.** Ingestion parses an upload in memory and writes chunks
and embeddings to the database; the original bytes are never written to disk
and never read back. A `storage_path` column existed on the model for a store-
the-original feature that was never built; migration `0002` drops it.
That makes the service stateless apart from Postgres, so it moves to any host
that runs a container -- and it is why the disk-persistence argument you might
expect against serverless does not actually apply here.

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
- **Nothing on the container is worth keeping.** The service is stateless; the
  database holds everything, and the demo corpus is reproducible with
  `scripts/seed_demo.py`.

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
