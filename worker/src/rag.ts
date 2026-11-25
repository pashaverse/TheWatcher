export interface Env {
  AI: Ai;
  VECTORIZE: VectorizeIndex;
  DISCORD_PUBLIC_KEY: string;
  DISCORD_APPLICATION_ID: string;
  GROQ_API_KEY?: string;
  GROQ_MODELS?: string;
  WORKERS_AI_MODEL?: string;
  MIN_SCORE?: string;
}

export const EMBED_MODEL = "@cf/baai/bge-small-en-v1.5";
// bge models expect this prefix on queries (not on the stored passages)
const QUERY_PREFIX = "Represent this sentence for searching relevant passages: ";

export interface Source {
  n: number;
  title: string;
  url: string;
  text: string;
  score: number;
}

export const SYSTEM_PROMPT = `You are The Watcher, a Discord assistant that answers questions about Information Technology University (ITU), Lahore, for students and applicants.

Answer ONLY from the numbered context below, which comes from the ITU Student Handbook and the official ITU website.

How to answer:
- Lead with the direct answer. Be clear and friendly, no role-play.
- Keep it short: a few sentences or a tight bullet list, under about 200 words unless the question really needs more.
- Use Discord markdown (bold, bullets). No headings, no tables.
- Cite the context you used with its number in square brackets, e.g. [1] or [2][3].
- If the context only partly answers the question, give what it does say and say what's missing.
- If the context doesn't answer it, say you couldn't find it in the handbook or ITU website and suggest who to ask (Admissions office, Registrar, or the relevant department). Do not guess.
- Never invent fees, dates, deadlines, percentages or names. For fees and deadlines, add that they can change and should be confirmed on itu.edu.pk.
- If the question isn't about ITU or student life there, briefly say that's outside what you can help with.`;

/** Embed the question and pull the most relevant chunks, grouped by page. */
export async function retrieve(env: Env, question: string): Promise<Source[]> {
  const emb = (await env.AI.run(EMBED_MODEL as any, { text: [QUERY_PREFIX + question] })) as unknown as { data: number[][] };
  const res = await env.VECTORIZE.query(emb.data[0], { topK: 8, returnMetadata: "all" });
  const minScore = Number(env.MIN_SCORE ?? "0.45");

  const bySource = new Map<string, Source>();
  let budget = 6500; // characters of context; keeps each question to ~2k tokens
  for (const m of res.matches) {
    if (m.score < minScore || budget <= 0) continue;
    const md = (m.metadata ?? {}) as Record<string, string>;
    const key = md.source || md.url || m.id;
    const text = (md.text ?? "").slice(0, budget);
    budget -= text.length;
    const existing = bySource.get(key);
    if (existing) existing.text += "\n…\n" + text;
    else bySource.set(key, { n: bySource.size + 1, title: md.title ?? "ITU", url: md.url ?? "", text, score: m.score });
  }
  return [...bySource.values()];
}

export function buildContext(sources: Source[]): string {
  if (!sources.length) return "(no relevant passages found)";
  return sources.map((s) => `[${s.n}] ${s.title}${s.url ? ` (${s.url})` : ""}\n${s.text}`).join("\n\n---\n\n");
}

type Msg = { role: "system" | "user"; content: string };

async function groq(env: Env, model: string, messages: Msg[], signal: AbortSignal): Promise<string> {
  const res = await fetch("https://api.groq.com/openai/v1/chat/completions", {
    method: "POST",
    headers: { authorization: `Bearer ${env.GROQ_API_KEY}`, "content-type": "application/json" },
    body: JSON.stringify({ model, messages, temperature: 0.2, max_tokens: 700 }),
    signal,
  });
  if (!res.ok) throw new Error(`groq ${model} ${res.status}: ${(await res.text()).slice(0, 200)}`);
  const data = (await res.json()) as any;
  const text = data?.choices?.[0]?.message?.content?.trim();
  if (!text) throw new Error(`groq ${model} empty response`);
  return text;
}

async function workersAI(env: Env, model: string, messages: Msg[]): Promise<string> {
  const out = (await env.AI.run(model as any, { messages, max_tokens: 700, temperature: 0.2 } as any)) as any;
  const text = (typeof out?.response === "string" ? out.response : "").trim();
  if (!text) throw new Error(`workers-ai ${model} empty response`);
  return text;
}

/**
 * Ask the models in order until one answers. Groq's free tier has per-model
 * daily limits, so when one model runs out the next one picks up, and Workers
 * AI is the final backstop.
 */
export async function generate(env: Env, messages: Msg[], deadline: number): Promise<{ text: string; model: string }> {
  const errors: string[] = [];
  const groqModels = env.GROQ_API_KEY
    ? (env.GROQ_MODELS ?? "llama-3.3-70b-versatile").split(",").map((s) => s.trim()).filter(Boolean)
    : [];

  for (const model of groqModels) {
    const left = deadline - Date.now();
    if (left < 4000) break; // leave time for the backstop
    try {
      const text = await groq(env, model, messages, AbortSignal.timeout(Math.min(10000, left - 3000)));
      return { text, model };
    } catch (e) {
      errors.push(String(e));
    }
  }
  const wa = env.WORKERS_AI_MODEL ?? "@cf/meta/llama-3.3-70b-instruct-fp8-fast";
  try {
    return { text: await workersAI(env, wa, messages), model: wa };
  } catch (e) {
    errors.push(String(e));
  }
  throw new Error("all models failed: " + errors.join(" | "));
}

export function citedSources(answer: string, sources: Source[]): Source[] {
  const cited = new Set([...answer.matchAll(/\[(\d+)\]/g)].map((m) => Number(m[1])));
  const picked = sources.filter((s) => cited.has(s.n));
  return picked.length ? picked : sources.slice(0, 2);
}
