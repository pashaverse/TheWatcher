import { EPHEMERAL, InteractionType, ResponseType, editOriginal, json, option, verifySignature } from "./discord";
import { Env, SYSTEM_PROMPT, Source, buildContext, citedSources, generate, retrieve } from "./rag";

const COLOR = 0x2f7d32;
const DISCLAIMER = "From the ITU handbook & website. Confirm fees and deadlines on itu.edu.pk.";

const HELP = [
  "**The Watcher** answers questions about ITU using the Student Handbook and the official ITU website.",
  "",
  "**Commands**",
  "`/ask question:` ask anything about ITU (admissions, fees, scholarships, grading, rules…)",
  "`/ask question: private:True` only you will see the answer",
  "`/help` show this message",
  "",
  "**Tips**",
  "• Ask one specific thing at a time: *\"What's the minimum CGPA to avoid probation?\"*",
  "• Answers list their sources. If something matters (fees, deadlines), double-check the linked page.",
].join("\n");

export default {
  async fetch(req: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(req.url);
    if (req.method === "GET" && url.pathname === "/health") return health(env);
    if (req.method === "GET") return new Response("The Watcher is online.", { status: 200 });
    if (req.method !== "POST" || !["/", "/interactions"].includes(url.pathname)) {
      return new Response("Not found", { status: 404 });
    }

    const body = await req.text();
    if (!(await verifySignature(req, body, env.DISCORD_PUBLIC_KEY))) {
      return new Response("Bad request signature", { status: 401 });
    }

    const interaction = JSON.parse(body);
    if (interaction.type === InteractionType.PING) return json({ type: ResponseType.PONG });

    if (interaction.type === InteractionType.COMMAND) {
      const name: string = interaction.data?.name;

      if (name === "help") {
        return json({ type: ResponseType.MESSAGE, data: { content: HELP, flags: EPHEMERAL } });
      }

      if (name === "ask" || name === "watcher") {
        const question = (option(interaction, "question") ?? option(interaction, "query") ?? "").trim();
        const isPrivate = option<boolean>(interaction, "private") === true;
        if (question.length < 3) {
          return json({ type: ResponseType.MESSAGE, data: { content: "Please ask a question, e.g. `/ask question: When are mid-term exams?`", flags: EPHEMERAL } });
        }
        // Discord needs a reply within 3s: show "thinking…" now, finish in the background.
        ctx.waitUntil(answer(env, interaction, question.slice(0, 500)));
        return json({ type: ResponseType.DEFERRED_MESSAGE, data: isPrivate ? { flags: EPHEMERAL } : {} });
      }
    }

    return json({ type: ResponseType.MESSAGE, data: { content: "Unknown command.", flags: EPHEMERAL } });
  },
} satisfies ExportedHandler<Env>;

/** End-to-end check used by the daily health workflow: can we embed and search? */
async function health(env: Env): Promise<Response> {
  try {
    const sources = await retrieve(env, "admission requirements for BS programs");
    const ok = sources.length > 0;
    return json({ ok, sources: sources.length, groq: Boolean(env.GROQ_API_KEY), discord: Boolean(env.DISCORD_PUBLIC_KEY) }, ok ? 200 : 503);
  } catch (e) {
    return json({ ok: false, error: String(e) }, 503);
  }
}

export async function answer(env: Env, interaction: any, question: string): Promise<void> {
  const started = Date.now();
  const deadline = started + 25_000; // background work is cut off ~30s after the reply
  const appId = interaction.application_id ?? env.DISCORD_APPLICATION_ID;
  try {
    const sources = await retrieve(env, question);
    const { text, model } = await generate(env, [
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user", content: `Context:\n${buildContext(sources)}\n\nQuestion: ${question}` },
    ], deadline);

    console.log(JSON.stringify({ q: question, model, sources: sources.length, ms: Date.now() - started }));
    await editOriginal(appId, interaction.token, { embeds: [answerEmbed(question, text, citedSources(text, sources), model)] });
  } catch (err) {
    console.error("answer failed", String(err));
    await editOriginal(appId, interaction.token, {
      embeds: [{
        color: 0xb3261e,
        description: "Sorry, I couldn't answer that right now. Please try again in a minute.",
      }],
    });
  }
}

function sourceLine(s: Source): string {
  const label = s.title.length > 80 ? s.title.slice(0, 77) + "…" : s.title;
  return s.url ? `[${s.n}] [${label}](${s.url})` : `[${s.n}] ${label}`;
}

export function answerEmbed(question: string, text: string, sources: Source[], model: string) {
  const embed: Record<string, unknown> = {
    color: COLOR,
    title: question.length > 250 ? question.slice(0, 247) + "…" : question,
    description: text.length > 4000 ? text.slice(0, 3990) + "…" : text,
    footer: { text: `${DISCLAIMER} · ${model.split("/").pop()}` },
  };
  if (sources.length) {
    embed.fields = [{ name: "Sources", value: sources.slice(0, 5).map(sourceLine).join("\n").slice(0, 1024) }];
  }
  return embed;
}
