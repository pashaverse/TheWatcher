const API = "https://discord.com/api/v10";

export const InteractionType = { PING: 1, COMMAND: 2, COMPONENT: 3, AUTOCOMPLETE: 4 } as const;
export const ResponseType = { PONG: 1, MESSAGE: 4, DEFERRED_MESSAGE: 5 } as const;
export const EPHEMERAL = 64;

function hexToBytes(hex: string): Uint8Array {
  const out = new Uint8Array(hex.length / 2);
  for (let i = 0; i < out.length; i++) out[i] = parseInt(hex.substr(i * 2, 2), 16);
  return out;
}

let cachedKey: { hex: string; key: CryptoKey } | undefined;

/** Discord signs every request with Ed25519; unsigned or forged requests must get a 401. */
export async function verifySignature(req: Request, body: string, publicKeyHex: string): Promise<boolean> {
  const sig = req.headers.get("X-Signature-Ed25519");
  const ts = req.headers.get("X-Signature-Timestamp");
  if (!sig || !ts || !/^[0-9a-f]{128}$/i.test(sig)) return false;
  try {
    if (cachedKey?.hex !== publicKeyHex) {
      const key = await crypto.subtle.importKey("raw", hexToBytes(publicKeyHex), { name: "Ed25519" }, false, ["verify"]);
      cachedKey = { hex: publicKeyHex, key };
    }
    return await crypto.subtle.verify(
      { name: "Ed25519" }, cachedKey.key, hexToBytes(sig), new TextEncoder().encode(ts + body),
    );
  } catch {
    return false;
  }
}

export function json(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), { status, headers: { "content-type": "application/json" } });
}

/** Replace the "thinking…" placeholder with the real answer. Retries once in plain text if Discord rejects it. */
export async function editOriginal(appId: string, token: string, payload: Record<string, unknown>): Promise<void> {
  const url = `${API}/webhooks/${appId}/${token}/messages/@original`;
  const send = (body: Record<string, unknown>) =>
    fetch(url, {
      method: "PATCH",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ allowed_mentions: { parse: [] }, ...body }),
    });

  let res = await send(payload);
  if (res.status === 429) {
    const wait = Number(res.headers.get("retry-after") ?? "1");
    await new Promise((r) => setTimeout(r, Math.min(wait, 5) * 1000));
    res = await send(payload);
  }
  if (!res.ok) {
    console.error("discord edit failed", res.status, await res.text().catch(() => ""));
    const embed = (payload.embeds as { description?: string }[] | undefined)?.[0];
    const text = embed?.description ?? String(payload.content ?? "Something went wrong.");
    await send({ content: text.slice(0, 1990), embeds: [] });
  }
}

/** Read a string/bool option from a slash command. */
export function option<T = string>(interaction: any, name: string): T | undefined {
  return interaction?.data?.options?.find((o: any) => o.name === name)?.value;
}
