// Registers the bot's slash commands with Discord (safe to run repeatedly).
// Usage: DISCORD_TOKEN=... DISCORD_APPLICATION_ID=... node scripts/register-commands.mjs
const token = process.env.DISCORD_TOKEN;
const appId = process.env.DISCORD_APPLICATION_ID;
if (!token || !appId) {
  console.error("DISCORD_TOKEN and DISCORD_APPLICATION_ID are required");
  process.exit(1);
}

// Usable in servers, in DMs with the bot, and (if users install it) anywhere.
const everywhere = { integration_types: [0, 1], contexts: [0, 1, 2] };

const questionOpt = (name) => ({
  name, type: 3, required: true, max_length: 500,
  description: "Your question about ITU (admissions, fees, scholarships, rules…)",
});
const privateOpt = { name: "private", type: 5, required: false, description: "Only you can see the answer" };

const commands = [
  { name: "ask", type: 1, description: "Ask The Watcher a question about ITU", options: [questionOpt("question"), privateOpt], ...everywhere },
  // kept so the old command people know still works
  { name: "watcher", type: 1, description: "Ask The Watcher a question about ITU", options: [questionOpt("query"), privateOpt], ...everywhere },
  { name: "help", type: 1, description: "How to use The Watcher", ...everywhere },
];

const res = await fetch(`https://discord.com/api/v10/applications/${appId}/commands`, {
  method: "PUT",
  headers: { Authorization: `Bot ${token}`, "Content-Type": "application/json" },
  body: JSON.stringify(commands),
});
if (!res.ok) {
  console.error(`Failed: ${res.status}`, await res.text());
  process.exit(1);
}
console.log(`Registered: ${(await res.json()).map((c) => "/" + c.name).join(", ")}`);
