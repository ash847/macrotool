# Evaluating MacroTool in Claude Desktop

MacroTool is an FX options structuring engine for macro PMs. In Claude Desktop it appears as one
tool, `ask_macrotool`. You describe a view in plain English; Claude hands it to MacroTool's own
analyst agent, which prices and ranks structures with its engine and replies with tables and a
short explanation. Claude is meant to show that reply as returned.

**Install:** see `install.sh` (about 5 minutes; you need a Mac, Claude Desktop, `uv`, and your own
Anthropic API key — each question costs a few cents on that key).

**Good to know before you start**
- Market data is a **static snapshot** in the repo (8 pairs: USDBRL, USDTRY, EURPLN, GBPUSD,
  EURUSD, USDJPY, USDMXN, USDCNH), so the numbers are illustrative, not live.
- Replies take roughly 6–30 seconds. One conversation is kept per Claude Desktop session; say
  "new trade" to start a different one.
- It is a prototype. Rough edges are expected; what we want to learn is whether the answers are
  *right and usable*.

## What to try (about 20 minutes)

Use these five views, then one of your own. Ask each as a fresh chat.

1. `I think USDJPY will trade in a range of 155-160 over the next two months`
2. `short USDTRY target 41 over a 3 month horizon`
3. `eurusd to move to 1.08 or below in 9m - what are the best trades and why?`
4. `usdjpy to 152 in 3m what are the best trades`
5. `usdtry to 48.00 in 3m what are the best 5 trades`

Then follow up on one of them: "why is #1 ranked first?", "compare #2 with a straight put",
"what's the main risk?", "no tails at all, rerun", "show a table with strikes, notional, premium
in %, loss budget and risk".

## What to check in each answer

| Check | Question to ask yourself |
| --- | --- |
| Direction | Does it read your view the right way round (e.g. "short USDTRY" = betting USD falls)? Is the with-carry / against-carry call right? |
| Numbers | Do spot, forward, vol, target distance and strikes look sensible and hang together? Anything that contradicts itself between tables and text? |
| Vocabulary | Plain and correct for a PM? Any term that confuses or is misused? |
| Structures | Are the ranked trades reasonable for the view? Does the reasoning for #1 make sense? Is anything obviously missing? |
| Risk | When it says a trade can or can't lose more than the premium, is that true? |
| Follow-ups | Does it stay on the same trade, and answer what you asked rather than re-running everything? |
| Fidelity | Does Claude show MacroTool's reply as returned, or reword/trim it, or answer from its own knowledge instead? (Say which.) |

## Compare with plain Claude (optional)

In one chat: *"<your view> First answer from your own knowledge without using any tools. Then ask
MacroTool and show its reply exactly as returned."*

Or generate a side-by-side report for the five views above (uses your API key, a few cents):

```bash
cd ~/macrotool-mcp && MACROTOOL_SECRETS=~/.config/macrotool/secrets.toml .venv/bin/python mcp_server/compare.py
```

Edit `mcp_server/compare_ideas.json` to test your own views.

## Feedback to send back

For each view: pair/target/tenor, what you asked, and a line each on **right / wrong / unclear**.
The most useful items are (1) any number or direction that is wrong, (2) any term you would not use,
(3) a trade you expected that was missing, (4) anywhere Claude changed or ignored MacroTool's
reply, and (5) whether you would use this in your workflow, and what would make it worth using.

If something fails to start, send the last 20 lines of
`~/Library/Logs/Claude/mcp-server-macrotool.log`.
