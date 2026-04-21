const state = {
  mode: "probability",
};

const els = {
  pInput: document.getElementById("pInput"),
  bInput: document.getElementById("bInput"),
  multiplierInput: document.getElementById("multiplierInput"),
  capEnabledInput: document.getElementById("capEnabledInput"),
  capInput: document.getElementById("capInput"),
  clampInput: document.getElementById("clampInput"),
  pOutput: document.getElementById("pOutput"),
  bOutput: document.getElementById("bOutput"),
  multiplierOutput: document.getElementById("multiplierOutput"),
  capOutput: document.getElementById("capOutput"),
  fullKellyValue: document.getElementById("fullKellyValue"),
  appliedKellyValue: document.getElementById("appliedKellyValue"),
  breakEvenValue: document.getElementById("breakEvenValue"),
  edgeCard: document.getElementById("edgeCard"),
  edgeValue: document.getElementById("edgeValue"),
  allocationLabel: document.getElementById("allocationLabel"),
  betSegment: document.getElementById("betSegment"),
  cashSegment: document.getElementById("cashSegment"),
  bankrollTrack: document.querySelector(".bankroll-track"),
  sensitivityChart: document.getElementById("sensitivityChart"),
  chartTitle: document.getElementById("chart-title"),
  modeProbability: document.getElementById("modeProbability"),
  modePayout: document.getElementById("modePayout"),
  comparisonBars: document.getElementById("comparisonBars"),
};

function kellyFraction(p, b) {
  const q = 1 - p;
  return (b * p - q) / b;
}

function breakEvenProbability(b) {
  return 1 / (b + 1);
}

function applySizing(fullKelly, multiplier, cap, clampNegative) {
  let applied = fullKelly * multiplier;
  if (clampNegative) applied = Math.max(0, applied);
  if (cap !== null && applied > cap) applied = cap;
  return applied;
}

function pct(value, digits = 2) {
  return `${(value * 100).toFixed(digits)}%`;
}

function multiple(value) {
  return `${value.toFixed(2)}x`;
}

function getInputs() {
  return {
    p: Number(els.pInput.value) / 100,
    b: Number(els.bInput.value),
    multiplier: Number(els.multiplierInput.value),
    cap: els.capEnabledInput.checked ? Number(els.capInput.value) / 100 : null,
    clampNegative: els.clampInput.checked,
  };
}

function setEdge(fullKelly) {
  els.edgeCard.classList.remove("positive", "negative", "neutral");

  if (fullKelly > 0.0001) {
    els.edgeCard.classList.add("positive");
    els.edgeValue.textContent = "Positive";
  } else if (fullKelly < -0.0001) {
    els.edgeCard.classList.add("negative");
    els.edgeValue.textContent = "Negative";
  } else {
    els.edgeCard.classList.add("neutral");
    els.edgeValue.textContent = "Neutral";
  }
}

function updateBankroll(appliedKelly) {
  const visibleBet = Math.max(0, Math.min(1, appliedKelly));
  els.betSegment.style.width = pct(visibleBet);
  els.allocationLabel.textContent = `${pct(appliedKelly)} bet`;
  els.bankrollTrack.classList.toggle("negative", appliedKelly < 0);
}

function comparisonData(fullKelly, appliedKelly) {
  return [
    ["Full Kelly", fullKelly],
    ["Half Kelly", fullKelly * 0.5],
    ["Quarter Kelly", fullKelly * 0.25],
    ["Current applied", appliedKelly],
  ];
}

function renderComparison(fullKelly, appliedKelly) {
  const rows = comparisonData(fullKelly, appliedKelly);
  const maxMagnitude = Math.max(0.01, ...rows.map(([, value]) => Math.abs(value)));

  els.comparisonBars.innerHTML = rows
    .map(([label, value]) => {
      const width = Math.min(100, Math.abs(value / maxMagnitude) * 100);
      const negativeClass = value < 0 ? " negative" : "";
      return `
        <div class="comparison-row">
          <span>${label}</span>
          <div class="mini-track">
            <div class="mini-fill${negativeClass}" style="width: ${width}%"></div>
          </div>
          <strong>${pct(value)}</strong>
        </div>
      `;
    })
    .join("");
}

function chartPoint(value, min, max, size, offset = 0) {
  return offset + ((value - min) / (max - min)) * size;
}

function renderChart(inputs, fullKelly, appliedKelly) {
  const width = 760;
  const height = 330;
  const margin = { top: 18, right: 24, bottom: 42, left: 56 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;
  const samples = 121;
  const isProbability = state.mode === "probability";

  const xMin = isProbability ? 0 : 0.1;
  const xMax = isProbability ? 1 : 10;
  const currentX = isProbability ? inputs.p : inputs.b;
  const values = Array.from({ length: samples }, (_, index) => {
    const t = index / (samples - 1);
    const x = xMin + (xMax - xMin) * t;
    const rawKelly = isProbability
      ? kellyFraction(x, inputs.b)
      : kellyFraction(inputs.p, x);
    return { x, y: rawKelly };
  });

  const yValues = values.map((point) => point.y).concat([0, fullKelly, appliedKelly]);
  const rawYMin = Math.min(...yValues);
  const rawYMax = Math.max(...yValues);
  const padding = Math.max(0.05, (rawYMax - rawYMin) * 0.12);
  const yMin = Math.min(-0.05, rawYMin - padding);
  const yMax = Math.max(0.05, rawYMax + padding);

  const xScale = (x) => chartPoint(x, xMin, xMax, innerWidth, margin.left);
  const yScale = (y) => margin.top + innerHeight - chartPoint(y, yMin, yMax, innerHeight);
  const linePath = values
    .map((point, index) => `${index === 0 ? "M" : "L"} ${xScale(point.x).toFixed(2)} ${yScale(point.y).toFixed(2)}`)
    .join(" ");
  const zeroY = yScale(0);
  const markerX = xScale(currentX);
  const markerY = yScale(fullKelly);
  const xLabel = isProbability ? "Win probability p" : "Payout multiple b";
  const markerLabel = isProbability ? pct(inputs.p, 1) : multiple(inputs.b);

  els.chartTitle.textContent = isProbability ? "Kelly vs Win Probability" : "Kelly vs Payout Multiple";

  els.sensitivityChart.setAttribute("viewBox", `0 0 ${width} ${height}`);
  els.sensitivityChart.innerHTML = `
    <rect x="0" y="0" width="${width}" height="${height}" rx="8" fill="#fbfcfa"></rect>
    <line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${height - margin.bottom}" stroke="#cfd8d2"></line>
    <line x1="${margin.left}" y1="${height - margin.bottom}" x2="${width - margin.right}" y2="${height - margin.bottom}" stroke="#cfd8d2"></line>
    <line x1="${margin.left}" y1="${zeroY}" x2="${width - margin.right}" y2="${zeroY}" stroke="#87928d" stroke-dasharray="5 5"></line>
    <text x="${margin.left - 10}" y="${yScale(yMax)}" text-anchor="end" dominant-baseline="middle" fill="#64716d" font-size="12">${pct(yMax, 0)}</text>
    <text x="${margin.left - 10}" y="${zeroY}" text-anchor="end" dominant-baseline="middle" fill="#64716d" font-size="12">0%</text>
    <text x="${margin.left - 10}" y="${yScale(yMin)}" text-anchor="end" dominant-baseline="middle" fill="#64716d" font-size="12">${pct(yMin, 0)}</text>
    <text x="${margin.left}" y="${height - 14}" fill="#64716d" font-size="12">${isProbability ? "0%" : "0.1x"}</text>
    <text x="${width - margin.right}" y="${height - 14}" text-anchor="end" fill="#64716d" font-size="12">${isProbability ? "100%" : "10x"}</text>
    <text x="${width / 2}" y="${height - 14}" text-anchor="middle" fill="#17211f" font-size="13" font-weight="800">${xLabel}</text>
    <text x="18" y="${height / 2}" transform="rotate(-90 18 ${height / 2})" text-anchor="middle" fill="#17211f" font-size="13" font-weight="800">Full Kelly fraction</text>
    <path d="${linePath}" fill="none" stroke="#0f8b6f" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"></path>
    <line x1="${markerX}" y1="${height - margin.bottom}" x2="${markerX}" y2="${markerY}" stroke="#17211f" stroke-width="2" stroke-dasharray="4 4"></line>
    <circle cx="${markerX}" cy="${markerY}" r="7" fill="#17211f"></circle>
    <circle cx="${markerX}" cy="${markerY}" r="3" fill="#ffffff"></circle>
    <text x="${Math.min(width - 90, Math.max(margin.left + 8, markerX + 12))}" y="${Math.max(30, markerY - 12)}" fill="#17211f" font-size="13" font-weight="800">${markerLabel}, ${pct(fullKelly)}</text>
  `;
}

function updateMode(mode) {
  state.mode = mode;
  els.modeProbability.classList.toggle("active", mode === "probability");
  els.modePayout.classList.toggle("active", mode === "payout");
  update();
}

function update() {
  const inputs = getInputs();
  const fullKelly = kellyFraction(inputs.p, inputs.b);
  const appliedKelly = applySizing(fullKelly, inputs.multiplier, inputs.cap, inputs.clampNegative);

  els.pOutput.textContent = pct(inputs.p, 1);
  els.bOutput.textContent = multiple(inputs.b);
  els.multiplierOutput.textContent = multiple(inputs.multiplier);
  els.capOutput.textContent = pct(Number(els.capInput.value) / 100, 1);
  els.capInput.disabled = !els.capEnabledInput.checked;

  els.fullKellyValue.textContent = pct(fullKelly);
  els.appliedKellyValue.textContent = pct(appliedKelly);
  els.breakEvenValue.textContent = pct(breakEvenProbability(inputs.b));

  setEdge(fullKelly);
  updateBankroll(appliedKelly);
  renderComparison(fullKelly, appliedKelly);
  renderChart(inputs, fullKelly, appliedKelly);
}

[
  els.pInput,
  els.bInput,
  els.multiplierInput,
  els.capEnabledInput,
  els.capInput,
  els.clampInput,
].forEach((input) => input.addEventListener("input", update));

els.modeProbability.addEventListener("click", () => updateMode("probability"));
els.modePayout.addEventListener("click", () => updateMode("payout"));

update();
