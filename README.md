# The Watcher

A Discord bot for the unofficial ITU server that answers admission and university questions from the **ITU Student Handbook** and the **official ITU website**. Inspired by The Watcher from the Marvel universe. Made for fun :p

Use `/ask` in any channel or in DMs with the bot. Answers are short, list their sources, and the bot says plainly when it doesn't know.

## How it works

```
Discord ──/ask──▶ Cloudflare Worker ──▶ Vectorize (search) ──▶ Groq / Workers AI (answer) ──▶ reply in Discord
                                          ▲
GitHub Actions, weekly ── crawl itu.edu.pk + handbook ── embed changed pages ──┘
```

| Piece | Where | Cost |
|---|---|---|
| Bot (`worker/`) | Cloudflare Workers: always on, no cold starts | free (100k requests/day) |
| Search index | Cloudflare Vectorize, ~1–2k chunks | free (5M stored dimensions) |
| Embeddings | Workers AI `bge-small-en-v1.5` | free (10k neurons/day) |
| Answers | Groq: Llama 3.3 70B → Llama 4 Scout → Llama 3.1 8B, then Workers AI as the backstop | free tiers |
| Weekly refresh, deploys, health check | GitHub Actions | free for public repos |

**Built to stay up**
- If one model hits its daily limit or goes down, the next one answers automatically.
- If the ITU site is down during a refresh, the old data is kept instead of being wiped.
- A daily health check runs a real search and emails you (via GitHub) if it fails.
- The weekly refresh commits a small status file, so GitHub never pauses the schedule.

## One-time setup (~15 minutes)

### 1. Cloudflare (free account)
1. Sign up at dash.cloudflare.com, open **Workers & Pages** once and pick your `*.workers.dev` subdomain.
2. Copy your **Account ID** (right sidebar of the Workers & Pages page).
3. **My Profile → API Tokens → Create Token → "Edit Cloudflare Workers" template**, then add two more permissions: **Account · Vectorize · Edit** and **Account · Workers AI · Read**. Create it and copy the token.

### 2. Discord
In the Developer Portal, open your existing app (ID `1457475133135524116`):
- **General Information → Public Key**
- **Bot token**: the `DISCORD_TOKEN` already in your local `.env`, or **Bot → Reset Token** for a new one.
- **Installation**: make sure both *Guild Install* and *User Install* are ticked and the `applications.commands` scope is included.

### 3. Groq
Create a free API key at console.groq.com/keys. (Optional: without it the bot still works on Workers AI alone.)

### 4. GitHub secrets
Repo → **Settings → Secrets and variables → Actions → New repository secret**:

| Secret | Value |
|---|---|
| `CF_ACCOUNT_ID` | Cloudflare account ID |
| `CF_API_TOKEN` | Cloudflare API token |
| `DISCORD_APPLICATION_ID` | `1457475133135524116` |
| `DISCORD_PUBLIC_KEY` | Discord public key |
| `DISCORD_TOKEN` | Discord bot token |
| `GROQ_API_KEY` | Groq key |

### 5. Run it
1. **Actions → Deploy bot → Run workflow.** This deploys the worker, registers `/ask` and `/help`, and points Discord at it. The run summary shows the worker URL.
2. **Actions → Refresh knowledge → Run workflow.** The first run crawls the site and uploads everything (a few minutes).
3. **Settings → Variables → New variable** `WORKER_URL` = the worker URL from step 1. This turns on the daily health check.
4. Try `/ask question: what scholarships does ITU offer?` in Discord.

After that it's hands-off: pushes to `worker/` redeploy automatically, and the knowledge base refreshes every Sunday.

## Everyday maintenance

- **Bot missed an important page?** Add its URL to `data/extra_urls.txt` and commit. A refresh runs automatically.
- **Wrong fact from the handbook?** `data/handbook.md` is plain text (half of it OCR'd from the PDF). Fix the line and commit.
- **New handbook PDF?** Replace `data/handbook.pdf`, run `python tools/extract_handbook.py data/handbook.pdf data/handbook.md` (needs `poppler-utils` and `tesseract`), commit.
- **Seeing what people ask / errors:** Cloudflare dashboard → Workers → thewatcher → Logs.
- **Change models or tone:** `GROQ_MODELS` in `worker/wrangler.toml`; the system prompt is in `worker/src/rag.ts`.

## Local development

```bash
cd worker && npm install && npm test          # bot tests (mocked Discord/Groq/Cloudflare)
pip install -r ingest/requirements.txt
python -m pytest ingest -q                    # refresh pipeline tests
python ingest/refresh.py --dry-run            # crawl + chunk without uploading
```
