'use strict';

// ── Configuration ─────────────────────────────────────────────────────────────

const DATA_BASE = (function () {
  const h = window.location.hostname;
  if (h === '' || h === 'localhost' || h === '127.0.0.1') {
    return '../data/processed/forecasts';
  }
  return './data/forecasts';
}());

const VARIABLE_META = {
  bnp_fastland:   { name: 'BNP Fastlands-Norge',  unit: '% år/år',
    description: 'Bruttonasjonalprodukt for fastlandsøkonomien. Viser den samlede verdiskapingen, ekskl. olje og gass.' },
  kpi:            { name: 'Konsumprisindeksen (KPI)', unit: '% år/år',
    description: 'Generell prisvekst. Måler endringen i prisen på en representativ kurv av varer og tjenester.' },
  kpi_jae:        { name: 'KPI justert (KPI-JAE)',  unit: '% år/år',
    description: 'KPI renset for avgiftsendringer og energipriser. Brukes av Norges Bank som mål på underliggende inflasjon.' },
  ledighet_aku:   { name: 'AKU-arbeidsledighet',   unit: '%',
    description: 'Andel av arbeidsstyrken som er uten jobb og aktivt søker arbeid (Arbeidskraftundersøkelsen, SSB).' },
  styringsrente:  { name: 'Styringsrenten',        unit: '%',
    description: 'Norges Banks viktigste pengepolitiske virkemiddel. Setter en gulvpris for renter i bankmarkedet.' },
  lonnsvekst:     { name: 'Lønnsvekst',            unit: '% år/år',
    description: 'Vekst i gjennomsnittlig årslønn for alle næringer. Viktig driver for inflasjon og kjøpekraft.' },
  boligprisvekst: { name: 'Boligprisvekst',        unit: '% år/år',
    description: 'Endring i SSBs boligprisindeks (bruktboligindeks, alle boligtyper, hele landet).' },
};

const MODEL_COLORS = {
  arima:       '#2563eb',
  bvar:        '#e87722',
  var:         '#16a34a',
  dfm:         '#7c3aed',
  arx:         '#dc2626',
  ml_baseline: '#6b7280',
};
const MODEL_LABELS = {
  arima:       'ARIMA',
  bvar:        'BVAR',
  var:         'VAR',
  dfm:         'DFM',
  arx:         'AR-X',
  ml_baseline: 'ML-baseline',
};
const MODEL_DESCRIPTIONS = {
  arima:       'Autoregressiv tidsseriemodell med integrering og glidende gjennomsnitt',
  bvar:        'Bayesiansk VAR med Minnesota-prior (shrinkage mot random walk)',
  var:         'Vektorautoregressiv modell – fanger dynamikk mellom flere variabler',
  dfm:         'Dynamisk faktormodell – komprimerer informasjon fra mange serier',
  arx:         'Autoregressiv modell med eksogene variabler (oljepris, renter, valuta)',
  ml_baseline: 'Maskinlæringsbaseline (gradient boosting med kvantilregresjon)',
};

const HIST_COLOR   = '#64748b';
const ENSEMBLE_COLOR = '#1d4ed8';
const FAN_COLOR    = 'rgba(29, 78, 216, 0.10)';
const FAN_EDGE     = 'rgba(29, 78, 216, 0.30)';

const LAYOUT_BASE = {
  paper_bgcolor: '#ffffff',
  plot_bgcolor:  '#f8fafc',
  font: { family: 'system-ui,-apple-system,sans-serif', size: 12, color: '#1e293b' },
  margin: { t: 10, r: 20, b: 56, l: 56 },
  legend: { orientation: 'h', y: -0.22, font: { size: 11 } },
  xaxis: { showgrid: false, showline: true, linecolor: '#e2e8f0', zeroline: false },
  yaxis: { showgrid: true, gridcolor: '#f1f5f9', zeroline: true, zerolinecolor: '#cbd5e1', zerolinewidth: 1.5 },
  shapes: [],
};
const PLOTLY_CFG = { responsive: true, displayModeBar: false };

// ── State ─────────────────────────────────────────────────────────────────────

let activeVar = null;
let activeChart = 'ensemble'; // 'ensemble' | 'models'

// ── Boot ──────────────────────────────────────────────────────────────────────

async function init() {
  try {
    const manifest = await getJSON(`${DATA_BASE}/manifest.json`);
    document.getElementById('last-updated').textContent =
      `Sist oppdatert: ${manifest.run_date}`;
    buildSidebar(manifest.variables);
    if (manifest.variables.length > 0) selectVar(manifest.variables[0]);
  } catch {
    showError('Prognosedata er ikke tilgjengelig ennå. Pipelinen kjøres automatisk hver mandag.');
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
    btn.textContent   = meta.name;
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

// ── Chart toggle ──────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('fan-toggle')?.addEventListener('click', e => {
    const btn = e.target.closest('.toggle-btn');
    if (!btn) return;
    const chart = btn.dataset.chart;
    if (chart === activeChart) return;
    activeChart = chart;
    document.querySelectorAll('#fan-toggle .toggle-btn').forEach(b =>
      b.classList.toggle('active', b.dataset.chart === chart));
    document.getElementById('ensemble-chart').hidden = (chart !== 'ensemble');
    document.getElementById('model-chart').hidden    = (chart !== 'models');
  });
});

// ── Master render ─────────────────────────────────────────────────────────────

function renderAll(data) {
  const meta = VARIABLE_META[data.variable_id] || { name: data.variable_id, unit: '', description: '' };
  document.getElementById('var-title').textContent = meta.name;
  document.getElementById('var-unit').textContent  = meta.unit;

  renderKeyFigures(data, meta);
  renderFan(data, meta.unit);
  renderModelChart(data, meta.unit);
  renderAccuracy(data.evaluation || {}, data.ensemble?.weights || {});
  renderDisagreement(data.disagreement || []);

  document.getElementById('variable-panel').hidden = false;
}

// ── Key figures ───────────────────────────────────────────────────────────────

function renderKeyFigures(data, meta) {
  const container = document.getElementById('key-figures');
  container.innerHTML = '';

  const history = data.history || [];
  const last    = history[history.length - 1];
  const forecasts = data.ensemble?.forecasts || [];

  if (meta.description) {
    const desc = document.createElement('p');
    desc.className = 'var-description';
    desc.textContent = meta.description;
    container.appendChild(desc);
  }

  const kfRow = document.createElement('div');
  kfRow.className = 'kf-row';

  if (last) {
    kfRow.appendChild(makeKF(
      'Siste observasjon',
      `${last.value.toFixed(1)} ${meta.unit}`,
      last.date.slice(0, 4),
    ));
  }
  for (const fc of forecasts.slice(0, 3)) {
    kfRow.appendChild(makeKF(
      `Prognose ${fc.date.slice(0, 4)}`,
      `${fc.q50.toFixed(1)} ${meta.unit}`,
      `Spenn: ${fc.q10.toFixed(1)} – ${fc.q90.toFixed(1)}`,
    ));
  }
  container.appendChild(kfRow);
}

function makeKF(label, value, sub) {
  const el = document.createElement('div');
  el.className = 'kf-card';
  el.innerHTML = `<span class="kf-label">${label}</span>
                  <span class="kf-value">${value}</span>
                  <span class="kf-sub">${sub}</span>`;
  return el;
}

// ── Ensemble fan chart (history + forecast) ───────────────────────────────────

function renderFan(data, unit) {
  const history   = data.history   || [];
  const forecasts = data.ensemble?.forecasts || [];
  if (!forecasts.length) return;

  const lastHistDate = history.length ? history[history.length - 1].date : null;

  // Vertical separator between history and forecast
  const shapes = lastHistDate ? [{
    type: 'line', xref: 'x', yref: 'paper',
    x0: lastHistDate, x1: lastHistDate, y0: 0, y1: 1,
    line: { color: '#94a3b8', width: 1, dash: 'dot' },
  }] : [];

  const hDates  = history.map(r => r.date);
  const hValues = history.map(r => r.value);
  const fDates  = forecasts.map(r => r.date);
  const q10     = forecasts.map(r => r.q10);
  const q50     = forecasts.map(r => r.q50);
  const q90     = forecasts.map(r => r.q90);

  const traces = [
    // Fan – upper edge (invisible, fill target)
    { x: fDates, y: q90, mode: 'lines', line: { color: 'transparent' },
      showlegend: false, hoverinfo: 'skip' },
    // Fan fill q10→q90
    { x: fDates, y: q10, mode: 'lines', fill: 'tonexty', fillcolor: FAN_COLOR,
      line: { color: 'transparent' }, showlegend: false, hoverinfo: 'skip' },
    // q10/q90 dashed edges
    { x: fDates, y: q10, mode: 'lines', name: '10.–90. persentil',
      line: { color: FAN_EDGE, dash: 'dot', width: 1 },
      hovertemplate: `Nedre grense: %{y:.1f} ${unit}<extra></extra>` },
    { x: fDates, y: q90, mode: 'lines', showlegend: false,
      line: { color: FAN_EDGE, dash: 'dot', width: 1 },
      hovertemplate: `Øvre grense: %{y:.1f} ${unit}<extra></extra>` },
    // Historical actuals
    { x: hDates, y: hValues, mode: 'lines+markers', name: 'Faktiske observasjoner',
      line: { color: HIST_COLOR, width: 2 }, marker: { size: 5, color: HIST_COLOR },
      hovertemplate: `%{x|%Y}: %{y:.1f} ${unit}<extra>Faktisk</extra>` },
    // Ensemble median
    { x: fDates, y: q50, mode: 'lines+markers', name: 'Ensemble-prognose (median)',
      line: { color: ENSEMBLE_COLOR, width: 2.5 }, marker: { size: 7 },
      hovertemplate: `%{x|%Y}: %{y:.1f} ${unit}<extra>Prognose</extra>` },
  ];

  Plotly.react('ensemble-chart', traces,
    { ...LAYOUT_BASE, shapes,
      yaxis: { ...LAYOUT_BASE.yaxis, title: { text: unit, standoff: 8 } } },
    PLOTLY_CFG);
}

// ── Model comparison chart ─────────────────────────────────────────────────────

function renderModelChart(data, unit) {
  const history   = data.history   || [];
  const models    = data.models    || {};
  const evaluation = data.evaluation || {};

  const lastHistDate = history.length ? history[history.length - 1].date : null;
  const shapes = lastHistDate ? [{
    type: 'line', xref: 'x', yref: 'paper',
    x0: lastHistDate, x1: lastHistDate, y0: 0, y1: 1,
    line: { color: '#94a3b8', width: 1, dash: 'dot' },
  }] : [];

  const traces = [];

  // Historical actuals reference
  if (history.length) {
    traces.push({
      x: history.map(r => r.date), y: history.map(r => r.value),
      mode: 'lines', name: 'Faktiske observasjoner',
      line: { color: HIST_COLOR, width: 2 },
      hovertemplate: `%{x|%Y}: %{y:.1f} ${unit}<extra>Faktisk</extra>`,
    });
  }

  // Per-model: backtest history (dashed) + future forecast (solid)
  for (const [id, fc] of Object.entries(models)) {
    const color = MODEL_COLORS[id] || '#888';
    const label = MODEL_LABELS[id] || id;
    const bt    = evaluation[id]?.backtest || [];

    // Backtest predictions (dashed, same colour, thinner)
    if (bt.length) {
      traces.push({
        x: bt.map(r => r.date), y: bt.map(r => r.forecast),
        mode: 'lines', name: `${label} (historisk)`,
        line: { color, width: 1.2, dash: 'dash' },
        showlegend: false,
        hovertemplate: `${label} prognose: %{y:.1f} ${unit}<extra>Backtest</extra>`,
      });
    }

    // Future forecast (solid)
    traces.push({
      x: fc.map(r => r.date), y: fc.map(r => r.q50),
      mode: 'lines+markers', name: label,
      line: { color, width: 1.8 }, marker: { size: 5, color },
      hovertemplate: `${label}: %{y:.1f} ${unit}<extra></extra>`,
    });
  }

  Plotly.react('model-chart', traces,
    { ...LAYOUT_BASE, shapes,
      yaxis: { ...LAYOUT_BASE.yaxis, title: { text: unit, standoff: 8 } } },
    PLOTLY_CFG);
}

// ── Accuracy table ────────────────────────────────────────────────────────────

function renderAccuracy(evaluation, weights) {
  const tbody = document.querySelector('#accuracy-table tbody');
  tbody.innerHTML = '';

  // Sort by RMSE ascending (best first), NaN last
  const entries = Object.entries(evaluation).sort((a, b) => {
    const ra = a[1].rmse ?? Infinity;
    const rb = b[1].rmse ?? Infinity;
    return ra - rb;
  });

  if (!entries.length) {
    const tr = document.createElement('tr');
    const td = document.createElement('td');
    td.colSpan = 6;
    td.textContent = 'Ingen backtesting-data tilgjengelig.';
    td.style.color = '#94a3b8';
    tr.appendChild(td);
    tbody.appendChild(tr);
    return;
  }

  const rmseValues = entries.map(([, ev]) => ev.rmse).filter(v => v != null && isFinite(v));
  const minRmse    = Math.min(...rmseValues);

  for (const [id, ev] of entries) {
    const tr = document.createElement('tr');
    const isBest = ev.rmse != null && ev.rmse === minRmse;
    if (isBest) tr.classList.add('row-best');

    const w    = weights[id] ?? null;
    const pct  = w != null ? (w * 100).toFixed(1) : '–';
    const barW = w != null ? Math.round(w * 120) : 0;

    const r2Val = ev.r2;
    const r2Str = r2Val != null ? r2Val.toFixed(2) : '–';
    const r2Class = r2Val != null && r2Val < 0 ? 'cell-warn' : '';

    setTd(tr, MODEL_LABELS[id] || id, 'cell-model');
    setTd(tr, ev.rmse != null ? ev.rmse.toFixed(2) : '–', isBest ? 'cell-best' : '');
    setTd(tr, ev.mae  != null ? ev.mae.toFixed(2)  : '–', '');
    setTd(tr, r2Str, r2Class);
    setTd(tr, ev.n_obs ?? '–', 'cell-muted');

    const tdW = document.createElement('td');
    tdW.innerHTML = w != null
      ? `<span class="weight-bar" style="width:${barW}px"></span>${pct}&nbsp;%`
      : '–';
    tr.appendChild(tdW);

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
    setTd(tr, dr.date ? dr.date.slice(0, 4) : dr.horizon_year);
    setTd(tr, dr.ensemble_q50.toFixed(2));
    setTd(tr, dr.spread.toFixed(2));
    setTd(tr, dr.std.toFixed(2));
    const flagTd = document.createElement('td');
    if (dr.high_disagreement) {
      flagTd.innerHTML = `<span class="flag-badge" title="Modellene er spesielt uenige for dette året">!</span>`;
    }
    tr.appendChild(flagTd);
    tbody.appendChild(tr);
  }
}

// ── Key figures ───────────────────────────────────────────────────────────────

// (renderKeyFigures is defined above near the top)

// ── Helpers ───────────────────────────────────────────────────────────────────

function setTd(tr, text, className) {
  const td = document.createElement('td');
  td.textContent = text;
  if (className) td.className = className;
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
