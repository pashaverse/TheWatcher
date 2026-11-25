import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import worker from "../src/index";

let privateKey: CryptoKey;
let publicHex: string;
const toHex = (b: ArrayBuffer) => [...new Uint8Array(b)].map((x) => x.toString(16).padStart(2, "0")).join("");

beforeAll(async () => {
  const kp = (await crypto.subtle.generateKey({ name: "Ed25519" }, true, ["sign", "verify"])) as CryptoKeyPair;
  privateKey = kp.privateKey;
  publicHex = toHex(await crypto.subtle.exportKey("raw", kp.publicKey));
});

async function signed(body: unknown, tamper = false): Promise<Request> {
  const text = JSON.stringify(body);
  const ts = "1700000000";
  const sig = toHex(await crypto.subtle.sign({ name: "Ed25519" }, privateKey, new TextEncoder().encode(ts + text)));
  return new Request("https://bot.example/interactions", {
    method: "POST",
    headers: { "X-Signature-Ed25519": tamper ? sig.replace(/^./, sig[0] === "a" ? "b" : "a") : sig, "X-Signature-Timestamp": ts },
    body: text,
  });
}

const matches = [
  { id: "1", score: 0.82, metadata: { source: "handbook:p10", title: "ITU Student Handbook, page 10", url: "", text: "Minimum passing grade for BS degree is D." } },
  { id: "2", score: 0.71, metadata: { source: "https://itu.edu.pk/fee/", title: "Fee Structure", url: "https://itu.edu.pk/fee/", text: "BS CS semester fee 165,000." } },
  { id: "3", score: 0.2, metadata: { source: "x", title: "Irrelevant", url: "", text: "noise" } },
];

function makeEnv(over: Record<string, unknown> = {}) {
  return {
    DISCORD_PUBLIC_KEY: publicHex,
    DISCORD_APPLICATION_ID: "app1",
    GROQ_API_KEY: "gsk_test",
    GROQ_MODELS: "model-a,model-b",
    AI: { run: vi.fn(async (model: string) => model.includes("bge") ? { data: [[0.1, 0.2]] } : { response: "Backstop answer [1]" }) },
    VECTORIZE: { query: vi.fn(async () => ({ matches })) },
    ...over,
  } as any;
}

function makeCtx() {
  const tasks: Promise<unknown>[] = [];
  return { ctx: { waitUntil: (p: Promise<unknown>) => tasks.push(p), passThroughOnException() {} } as any, tasks };
}

const ask = (q: string, extra: object[] = []) => ({
  type: 2, token: "tok", application_id: "app1",
  data: { name: "ask", options: [{ name: "question", type: 3, value: q }, ...extra] },
});

let fetchMock: ReturnType<typeof vi.fn>;
let discordEdits: any[];
let groqCalls: string[];
let groqBehaviour: Record<string, () => Response>;

beforeEach(() => {
  discordEdits = [];
  groqCalls = [];
  groqBehaviour = {};
  fetchMock = vi.fn(async (url: string, init: RequestInit) => {
    if (url.includes("groq.com")) {
      const model = JSON.parse(String(init.body)).model;
      groqCalls.push(model);
      if (groqBehaviour[model]) return groqBehaviour[model]();
      return new Response(JSON.stringify({ choices: [{ message: { content: `The minimum passing grade for BS is **D** [1]. (${model})` } }] }));
    }
    if (url.includes("discord.com")) {
      discordEdits.push(JSON.parse(String(init.body)));
      return new Response("{}", { status: 200 });
    }
    throw new Error("unexpected fetch " + url);
  });
  vi.stubGlobal("fetch", fetchMock);
});

describe("discord plumbing", () => {
  it("answers GET health check", async () => {
    const res = await worker.fetch(new Request("https://bot.example/"), makeEnv(), makeCtx().ctx);
    expect(res.status).toBe(200);
  });

  it("/health reports search working, and 503 when the index is empty", async () => {
    const ok = await worker.fetch(new Request("https://bot.example/health"), makeEnv(), makeCtx().ctx);
    expect(ok.status).toBe(200);
    expect(((await ok.json()) as any).sources).toBe(2);
    const empty = makeEnv({ VECTORIZE: { query: vi.fn(async () => ({ matches: [] })) } });
    const bad = await worker.fetch(new Request("https://bot.example/health"), empty, makeCtx().ctx);
    expect(bad.status).toBe(503);
  });

  it("rejects bad signatures", async () => {
    const res = await worker.fetch(await signed({ type: 1 }, true), makeEnv(), makeCtx().ctx);
    expect(res.status).toBe(401);
    const res2 = await worker.fetch(new Request("https://bot.example/interactions", { method: "POST", body: "{}" }), makeEnv(), makeCtx().ctx);
    expect(res2.status).toBe(401);
  });

  it("responds to PING", async () => {
    const res = await worker.fetch(await signed({ type: 1 }), makeEnv(), makeCtx().ctx);
    expect(await res.json()).toEqual({ type: 1 });
  });

  it("/help is ephemeral and immediate", async () => {
    const res = await worker.fetch(await signed({ type: 2, data: { name: "help" } }), makeEnv(), makeCtx().ctx);
    const body: any = await res.json();
    expect(body.type).toBe(4);
    expect(body.data.flags).toBe(64);
  });
});

describe("/ask", () => {
  it("defers, then edits in an embed with cited sources", async () => {
    const { ctx, tasks } = makeCtx();
    const env = makeEnv();
    const res = await worker.fetch(await signed(ask("What is the passing grade for BS?")), env, ctx);
    expect(await res.json()).toEqual({ type: 5, data: {} });
    await Promise.all(tasks);

    expect(groqCalls).toEqual(["model-a"]);
    const sys = JSON.parse(String(fetchMock.mock.calls[0][1].body)).messages;
    expect(sys[1].content).toContain("[1] ITU Student Handbook, page 10");
    expect(sys[1].content).toContain("[2] Fee Structure");
    expect(sys[1].content).not.toContain("Irrelevant"); // below score threshold

    const embed = discordEdits[0].embeds[0];
    expect(embed.description).toContain("**D** [1]");
    expect(embed.fields[0].value).toBe("[1] ITU Student Handbook, page 10"); // only cited source listed
    expect(discordEdits[0].allowed_mentions).toEqual({ parse: [] });
  });

  it("private option makes the reply ephemeral; legacy /watcher still works", async () => {
    const { ctx, tasks } = makeCtx();
    const res = await worker.fetch(await signed(ask("fees?", [{ name: "private", type: 5, value: true }])), makeEnv(), ctx);
    expect(await res.json()).toEqual({ type: 5, data: { flags: 64 } });
    await Promise.all(tasks);

    const legacy = { type: 2, token: "t", application_id: "app1", data: { name: "watcher", options: [{ name: "query", value: "dean?" }] } };
    const c2 = makeCtx();
    const res2 = await worker.fetch(await signed(legacy), makeEnv(), c2.ctx);
    expect((await res2.json() as any).type).toBe(5);
  });

  it("falls through the model chain when one is rate limited", async () => {
    groqBehaviour["model-a"] = () => new Response("rate limited", { status: 429 });
    const { ctx, tasks } = makeCtx();
    await worker.fetch(await signed(ask("grading?")), makeEnv(), ctx);
    await Promise.all(tasks);
    expect(groqCalls).toEqual(["model-a", "model-b"]);
    expect(discordEdits[0].embeds[0].footer.text).toContain("model-b");
  });

  it("uses Workers AI when every Groq model fails", async () => {
    groqBehaviour["model-a"] = () => new Response("down", { status: 500 });
    groqBehaviour["model-b"] = () => new Response("down", { status: 503 });
    const { ctx, tasks } = makeCtx();
    await worker.fetch(await signed(ask("grading?")), makeEnv(), ctx);
    await Promise.all(tasks);
    expect(discordEdits[0].embeds[0].description).toBe("Backstop answer [1]");
  });

  it("works with no Groq key at all", async () => {
    const { ctx, tasks } = makeCtx();
    await worker.fetch(await signed(ask("grading?")), makeEnv({ GROQ_API_KEY: undefined }), ctx);
    await Promise.all(tasks);
    expect(groqCalls).toEqual([]);
    expect(discordEdits[0].embeds[0].description).toBe("Backstop answer [1]");
  });

  it("posts a friendly error if search breaks, instead of leaving 'thinking…' forever", async () => {
    const env = makeEnv({ VECTORIZE: { query: vi.fn(async () => { throw new Error("index missing"); }) } });
    const { ctx, tasks } = makeCtx();
    await worker.fetch(await signed(ask("grading?")), env, ctx);
    await Promise.all(tasks);
    expect(discordEdits[0].embeds[0].description).toMatch(/couldn't answer/);
  });

  it("truncates very long answers to fit Discord limits", async () => {
    groqBehaviour["model-a"] = () => new Response(JSON.stringify({ choices: [{ message: { content: "x".repeat(6000) } }] }));
    const { ctx, tasks } = makeCtx();
    await worker.fetch(await signed(ask("long?")), makeEnv(), ctx);
    await Promise.all(tasks);
    expect(discordEdits[0].embeds[0].description.length).toBeLessThanOrEqual(4000);
  });

  it("rejects an empty question without calling any model", async () => {
    const res = await worker.fetch(await signed(ask("  ")), makeEnv(), makeCtx().ctx);
    const body: any = await res.json();
    expect(body.type).toBe(4);
    expect(groqCalls).toEqual([]);
  });
});
