'use strict';

// ── Configuration ─────────────────────────────────────────────────────────────

// Local dev: serve from repo root (python -m http.server from /SMART).
// GitHub Pages: index.html is at site root, data is at data/forecasts/.
const DATA_BASE = (function () {
  const h = window.location.hostname;
  if (h === '' || h === 'localhost' || h === '127.0.0.1') {
    return '../data/processed/forecasts';
  }
  return './data/forecasts';
}());

const VARIABLE_META = {
  bnp_fastland:   { name: 'BNP Fastlands-Norge',   unit: '% år/år' },
  kpi:            { name: 'KPI',                    unit: '% år/år' },
  kpi_jae:        { name: 'KPI-JAE',                unit: '% år/år' },
  ledighet_aku:   { name: 'AKU-arbeidsledighet',    unit: '%'       },
  styringsrente:  { name: 'Styringsrenten',         unit: '%'       },
  lonnsvekst:     { name: 'Lønnsvekst',             unit: '% år/år' },
  boligprisvekst: { name: 'Boligprisvekst',         unit: '% år/år' },
};

const MODEL_COLORS = {
  arima:       '#1a4b8c',
  bvar:        '#e87722',
  var:         '#2e9e44',
  dfm:         '#8b2252',
  arx:         '#c0392b',
  ml_baseline: '#5b5b5b',
};
const MODEL_LABELS = {
  arima:       'ARIMA',
  bvar:        'BVAR',
  var:         'VAR',
  dfm:         'DFM',
  arx:         'AR-X',
  ml_baseline: 'ML-baseline',
};

const ENSEMBLE_COLOR = '#1a4b8c';
const FAN_COLOR      = 'rgba(26, 75, 140, 0.12)';

const LAYOUT_BASE = {
  paper_bgcolor: '#ffffff',
  plot_bgcolor:  '#f8f9fb',
  font: { family: 'system-ui,-apple-system,sans-serif', size: 12, color: '#1a1e2e' },
  margin: { t: 20, r: 20, b: 56, l: 56 },
  legend: { orientation: 'h', y: -0.22 },
  xaxis: { showgrid: false, showline: true, linecolor: '#dde3ec' },
  yaxis: { showgrid: true, gridcolor: '#eef0f5', zeroline: true, zerolinecolor: '#c8cdd8' },
};
const PLOTLY_CFG = { responsive: true, displayModeBar: false };

// ── State ─────────────────────────────────────────────────────────────────────

let activeVar = null;

// ── Boot ──────────────────────────────────────────────────────────────────────

async function init() {
  try {
    const manifest = await getJSON(`${DATA_BASE}/manifest.json`);
    document.getElementById('last-updated').textContent =
      `Sist oppdatert: ${manifest.run_date}`;
    buildSidebar(manifest.variables);
    if (manifest.variables.length > 0) selectVar(manifest.variables[0]);
  } catch {
    showError('Prognosedata ikke tilgjengelig ennå. Pipelinen kjøres ukentlig.');
  }
}

// ── Sidebar ───────────────────────────────────────────────────────────────────

function buildSidebar(variables) {
  const ul = document.getElementById('variable-list');
  ul.innerHTML = '';
  for (const id of variables) {
    const meta = VARIABLE_META[id] || { name: id };
    const li   = document.createElement('li');
    const btn  = document.createElement('button');
    btn.textContent  = meta.name;
    btn.dataset.varId = id;
    btn.addEventListener('click', () => selectVar(id));
    li.appendChild(btn);
    ul.appendChild(li);
  }
}

function highlightSidebar(varId) {
  document.querySelectorAll('#variable-list button').forEach(b => {
    b.classList.toggle('active', b.dataset.varId === varId);
  });
}

// ── Variable selection ────────────────────────────────────────────────────────

async function selectVar(varId) {
  if (varId === activeVar) return;
  activeVar = varId;
  highlightSidebar(varId);
  hideError();
  setLoading(true);

  try {
    const data = await getJSON(`${DATA_BASE}/${varId}/latest.json`);
    renderAll(data);
    setLoading(false);
  } catch {
    setLoading(false);
    showError(`Ingen prognosedata for ${VARIABLE_META[varId]?.name ?? varId} ennå.`);
  }
}

// ── Master render ─────────────────────────────────────────────────────────────

function renderAll(data) {
  const meta = VARIABLE_META[data.variable_id] || { name: data.variable_id, unit: '' };
  document.getElementById('var-title').textContent = meta.name;
  document.getElementById('var-unit').textContent  = meta.unit;

  renderFan(data.ensemble.forecasts, meta.unit);
  renderModels(data.models, meta.unit);
  renderWeights(data.ensemble.weights);
  renderDisagreement(data.disagreement);

  document.getElementById('variable-panel').hidden = false;
}

// ── Ensemble fan chart ────────────────────────────────────────────────────────

function renderFan(forecasts, unit) {
  const dates = forecasts.map(r => r.date);
  const q10   = forecasts.map(r => r.q10);
  const q50   = forecasts.map(r => r.q50);
  const q90   = forecasts.map(r => r.q90);

  const traces = [
    // Invisible upper edge for fill reference
    { x: dates, y: q90, mode: 'lines', line: { color: 'transparent' },
      showlegend: false, hoverinfo: 'skip' },
    // Fan fill q10→q90
    { x: dates, y: q10, mode: 'lines', fill: 'tonexty', fillcolor: FAN_COLOR,
      line: { color: 'transparent' }, showlegend: false, hoverinfo: 'skip' },
    // q10 dashed
    { x: dates, y: q10, mode: 'lines', name: '10. persentil',
      line: { color: ENSEMBLE_COLOR, dash: 'dot', width: 1 },
      hovertemplate: `q10: %{y:.2f} ${unit}<extra></extra>` },
    // q90 dashed
    { x: dates, y: q90, mode: 'lines', name: '90. persentil',
      line: { color: ENSEMBLE_COLOR, dash: 'dot', width: 1 },
      hovertemplate: `q90: %{y:.2f} ${unit}<extra></extra>` },
    // q50 median line
    { x: dates, y: q50, mode: 'lines+markers', name: 'Ensemble median',
      line: { color: ENSEMBLE_COLOR, width: 2.5 }, marker: { size: 6 },
      hovertemplate: `Median: %{y:.2f} ${unit}<extra></extra>` },
  ];

  Plotly.react('ensemble-chart', traces,
    { ...LAYOUT_BASE,
      yaxis: { ...LAYOUT_BASE.yaxis, title: { text: unit, standoff: 8 } } },
    PLOTLY_CFG);
}

// ── Model comparison chart ────────────────────────────────────────────────────

function renderModels(models, unit) {
  const traces = Object.entries(models).map(([id, fc]) => ({
    x: fc.map(r => r.date),
    y: fc.map(r => r.q50),
    mode: 'lines+markers',
    name: MODEL_LABELS[id] || id,
    line: { color: MODEL_COLORS[id] || '#888', width: 1.8 },
    marker: { size: 5 },
    hovertemplate: `${MODEL_LABELS[id] || id}: %{y:.2f} ${unit}<extra></extra>`,
  }));

  Plotly.react('model-chart', traces,
    { ...LAYOUT_BASE,
      yaxis: { ...LAYOUT_BASE.yaxis, title: { text: unit, standoff: 8 } } },
    PLOTLY_CFG);
}

// ── Weights table ─────────────────────────────────────────────────────────────

function renderWeights(weights) {
  const tbody = document.querySelector('#weights-table tbody');
  tbody.innerHTML = '';
  const sorted = Object.entries(weights).sort((a, b) => b[1] - a[1]);
  for (const [id, w] of sorted) {
    const tr = document.createElement('tr');
    setTd(tr, MODEL_LABELS[id] || id);
    const pct  = (w * 100).toFixed(1);
    const barW = Math.min(pct * 1.2, 100);
    const td   = document.createElement('td');
    td.innerHTML =
      `<span class="weight-bar" style="width:${barW}px"></span>${pct} %`;
    tr.appendChild(td);
    tbody.appendChild(tr);
  }
}

// ── Disagreement table ────────────────────────────────────────────────────────

function renderDisagreement(disagreement) {
  const tbody = document.querySelector('#disagreement-table tbody');
  tbody.innerHTML = '';
  for (const dr of disagreement) {
    const tr = document.createElement('tr');
    if (dr.high_disagreement) tr.classList.add('row-warn');
    setTd(tr, dr.horizon_year);
    setTd(tr, dr.ensemble_q50.toFixed(2));
    setTd(tr, dr.spread.toFixed(2));
    setTd(tr, dr.std.toFixed(2));
    const flagTd = document.createElement('td');
    if (dr.high_disagreement) flagTd.innerHTML = '<span class="flag-badge">!</span>';
    tr.appendChild(flagTd);
    tbody.appendChild(tr);
  }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function setTd(tr, text) {
  const td = document.createElement('td');
  td.textContent = text;
  tr.appendChild(td);
}

async function getJSON(url) {
  const res = await fetch(url, { cache: 'no-cache' });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

function setLoading(show) {
  document.getElementById('loading-state').hidden = !show;
  if (show) document.getElementById('variable-panel').hidden = true;
}

function showError(msg) {
  document.getElementById('error-message').textContent = msg;
  document.getElementById('error-state').hidden = false;
  document.getElementById('loading-state').hidden = true;
}

function hideError() {
  document.getElementById('error-state').hidden = true;
}

// ── Start ─────────────────────────────────────────────────────────────────────

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
