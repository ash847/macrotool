# MacroTool — Concept Note

*An EM FX options structuring & sizing tool for macro-fund PMs, with a conversational layer.*

---

## In one line

A PM states a view in plain English; the tool computes — deterministically — which option
structures best express it, sizes them, and lets the PM **converse about the comparison**
rather than read a static report.

## The problem it solves

When a macro PM has a directional FX view, the hard part isn't the direction — it's the
*structuring*: vanilla or spread? capped or open tail? how much premium, what strikes, what
sizing, how does each choice behave if the move is slow vs sharp, with the carry or against
it? Today that lives in a structurer's head and a spreadsheet. The judgement is real but
slow, inconsistent, and hard to interrogate ("why that one and not the spread?").

## The core idea

Two parts, deliberately separated:

1. **A deterministic engine** does all the quantitative work. Given a view and live market
   data, it builds market state (carry, vol regime, distances to target), **scores and ranks
   candidate structures** against tunable domain rules, prices concrete variants (Black-76
   with a smile vol surface), **evaluates each across a scenario grid** (slow/fast/with/against
   the move), and sizes to a risk budget. Every number is computed here.

2. **A conversational agent** sits on top. The PM talks to it; it routes each question to the
   engine, then **narrates the results in a PM's language**. Crucially, the agent *never
   invents a number* — it can only relay what the engine computed. It picks which engine
   tools to run; the engine owns every figure.

The product's centre of gravity is the **scoring and the comparative power it gives** — the
ability to say not just "here's a 1×1.5 put spread" but "*here's why it beats the 1×1 for your
view: it holds up better on a slow path while giving up little if the move is sharp.*" The
agent is a lens onto that comparison, **not** a free-form pricing calculator.

## Why this design

- **Trust.** In a tool that suggests trades, a hallucinated strike or premium is
  unacceptable. By construction, the language model produces *prose, never numbers* — so the
  conversational flexibility never compromises numerical correctness.
- **Consistency & speed.** The structuring judgement is encoded once, in tunable rules and
  scenario weightings a domain expert can adjust without touching code, and applied uniformly
  in seconds.
- **Interrogability.** Because the engine evaluates every candidate across scenarios, the PM
  can ask the comparison questions a static recommendation can't answer.

## What's been built

- **The deterministic engine** — market-state computation, rule-based structure
  selection/ranking, smile-vol option pricing across the structure families (vanillas,
  call/put spreads, ratio spreads, seagulls, digitals, reverse-knock-outs), scenario
  evaluation, and Kelly-style sizing. Four EM/G10 pairs wired (USDBRL, USDTRY, EURPLN,
  GBPUSD), each with its own market character.
- **The conversational agent** — a tool-calling loop in the live app. A PM types a view
  ("long USDBRL, 3m, target 5.60"); the agent runs the full engine and replies with the
  market read plus the **top recommended structures, each spelled out leg by leg** (side,
  delta, strike, sized notional, premium, payoff, risk). Follow-ups re-run only what changed.
- **A first-class "product model."** Structures are now represented as explicit lists of legs
  — so every construction detail (the deltas, the ratio weights, the seagull's funding wing)
  is real data the agent reads off, rather than something it could mis-state. This was a
  deliberate engineering refactor to make the agent reliable *by construction*.

## The aim — where this is going

In priority order, and deliberately staged:

1. **Converse about the comparison, to a high standard.** Make the agent fluent in *why* the
   engine ranks structures as it does — the scenario-by-scenario trade-offs, not just the
   list. This is the product's anchor and the immediate focus.
2. **Make it demonstrably trustworthy.** Systematic observability and verification so every
   stated number is provably engine-sourced — holding a high reliability bar as the
   conversation gets richer.
3. **Explore within the considered set.** Let the PM ask for the *other* variants the engine
   already evaluated, or nudge a strike — staying inside what the engine scored, not inventing
   new trades.

Explicitly **later, if ever**: free-form "price me any structure" capability. The intent is a
**conversational analyst over a rigorous scoring engine**, not a chat-driven pricing
calculator — the discipline is staying close to the comparative scoring that makes the tool
valuable.

## Status, plainly

The engine and the conversational agent both work end-to-end in the live app today. Current
effort is on conversational quality and reliability — making the agent argue the engine's
comparison faithfully — before broadening what the PM can ask it to price.

*This is a working prototype on synthetic/representative market data, intended to prove the
structuring and conversational approach.*
