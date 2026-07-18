"use strict";
const $ = (id) => document.getElementById(id);
const CATS = ["TP", "FN", "FP", "TN"];
const WARD_TITLES = {
  user: "Ward Snapshot — My Model",
  tung: "Ward Snapshot — Tung Model",
  ensemble: "Ward Snapshot — Ensemble (50/50 blend)",
};
let state = { model: "user", hour: 24, selected: null, playing: false, timer: null, speed: 3000, cmp: null };

// ---------- init ----------
fetch("/api/config").then(r => r.json()).then(cfg => {
  state.cfg = cfg;
  document.querySelectorAll(".tab").forEach(t => t.addEventListener("click", () => switchTab(t)));
  $("hour").addEventListener("input", e => { setHour(+e.target.value); });
  $("btn-play").addEventListener("click", togglePlay);
  $("btn-restart").addEventListener("click", restart);
  $("speed").addEventListener("change", e => { state.speed = +e.target.value; if (state.playing) { stopPlay(); startPlay(); } });
  $("cmp-hour").addEventListener("input", e => renderCompareAt(+e.target.value));
  $("pick-septic").addEventListener("click", () => loadRandom(true));
  $("pick-any").addEventListener("click", () => loadRandom(false));
  updateHourDisplay();
  loadWard();
});

function setHour(h) {
  state.hour = h;
  $("hour").value = h;
  updateHourDisplay();
  loadWard();
  if (state.selected) loadDetail(state.selected);
}
function updateHourDisplay() { $("sim-hour").textContent = "Hour " + state.hour; }

function switchTab(tab) {
  document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
  tab.classList.add("active");
  const which = tab.dataset.tab;
  stopPlay();
  if (which === "compare") {
    $("ward-view").classList.add("hidden");
    $("compare-view").classList.remove("hidden");
    if (!state.cmp) loadRandom(true);
  } else {
    $("compare-view").classList.add("hidden");
    $("ward-view").classList.remove("hidden");
    state.model = tab.dataset.model;
    $("ward-title").textContent = WARD_TITLES[state.model];
    loadWard();
    if (state.selected) loadDetail(state.selected);
  }
}

// ---------- ward (tabs 1–3) ----------
function loadWard() {
  fetch(`/api/beds?model=${state.model}&hour=${state.hour}`).then(r => r.json()).then(beds => {
    CATS.forEach(c => $("beds-" + c).innerHTML = "");
    beds.forEach(b => $("beds-" + b.category).appendChild(bedCard(b)));
  });
}

function bedCard(b) {
  const el = document.createElement("div");
  const tier = b.tier || "LOW";
  el.className = "bed-card" + (b.is_discharged ? " discharged" : "") + (state.selected === b.id ? " active" : "");
  const crit = (b.criticality == null) ? "—" : b.criticality;
  const trendCls = b.trend || "steady";
  const trendSym = b.trend === "rising" ? "▲" : b.trend === "falling" ? "▼" : "→";
  const pred = b.is_risky
    ? `<div class="bed-prediction pred-sepsis">PREDICTS SEPSIS</div>`
    : `<div class="bed-prediction pred-safe">NO SEPSIS RISK</div>`;
  el.innerHTML = `
    <div class="bed-header"><span class="bed-id">Bed ${b.id}</span><span class="trend ${trendCls}">${trendSym}</span></div>
    ${pred}
    <div class="bed-crit tier-${tier}">${crit}<small>/100</small></div>
    <div class="tier-label tier-${tier}">${b.tung_unavailable ? "TUNG OFFLINE" : tier}</div>`;
  el.addEventListener("click", () => {
    state.selected = b.id;
    document.querySelectorAll(".bed-card").forEach(x => x.classList.remove("active"));
    el.classList.add("active");
    loadDetail(b.id);
  });
  return el;
}

function loadDetail(pid) {
  fetch(`/api/beds/${pid}?model=${state.model}&hour=${state.hour}`).then(r => r.json()).then(d => {
    $("detail-empty").classList.add("hidden");
    $("detail-content").classList.remove("hidden");
    $("d-pid").textContent = "Bed " + d.id + (d.category ? "  ·  " + d.category : "");
    const tier = d.tier || "—";
    $("d-tier").textContent = tier;
    $("d-tier").className = "badge tier-badge tier-" + tier;

    const banner = $("d-banner");
    banner.textContent = d.is_risky ? "PREDICTS SEPSIS" : "NO SEPSIS RISK";
    banner.className = "prediction-banner " + (d.is_risky ? "pred-sepsis" : "pred-safe");

    $("d-crit").textContent = d.criticality == null ? "—" : d.criticality;
    const tr = $("d-trend");
    tr.className = "trend-icon trend " + (d.trend || "steady");
    tr.textContent = d.trend === "rising" ? "▲" : d.trend === "falling" ? "▼" : "→";
    $("d-prob-label").textContent = state.model === "user" ? "Calibrated Risk" : "Risk Probability";
    $("d-prob").textContent = d.probability == null ? "—" : d.probability + "%";
    $("d-alert").innerHTML = d.is_risky
      ? `<span class="tier-CRITICAL">● ALARM</span>`
      : `<span class="tier-LOW">○ Clear</span>`;
    $("d-truth").innerHTML = d.sepsis_now
      ? `<span class="tier-CRITICAL">SEPSIS</span>`
      : (d.true_onset_hour ? `Onset h${d.true_onset_hour}` : `<span class="tier-LOW">None</span>`);

    renderDrivers("drv-vitals", (d.drivers || {}).vitals_labs);
    renderDrivers("drv-demo", (d.drivers || {}).demographics);
    renderDrivers("drv-others", (d.drivers || {}).others);
  });
}

function renderDrivers(elId, list) {
  const ul = $(elId);
  ul.innerHTML = "";
  if (!list || !list.length) {
    ul.innerHTML = `<li class="empty-driver">None</li>`;
    return;
  }
  list.forEach(dr => {
    const li = document.createElement("li");
    li.className = dr.direction === "↑" ? "up" : "down";
    const val = dr.value == null ? "" : ` (${dr.value})`;
    li.innerHTML = `<span>${dr.label}${val} ${dr.direction}</span>` +
      (dr.source ? `<span class="drv-src ${dr.source}">${dr.source}</span>` : "");
    ul.appendChild(li);
  });
}

// ---------- simulation ----------
function togglePlay() { state.playing ? stopPlay() : startPlay(); }
function startPlay() {
  state.playing = true;
  $("btn-play").textContent = "Pause";
  state.timer = setInterval(() => {
    if (state.hour >= 72) { stopPlay(); return; }
    setHour(state.hour + 1);
  }, state.speed);
}
function stopPlay() {
  state.playing = false;
  $("btn-play").textContent = "Play";
  if (state.timer) clearInterval(state.timer);
}
function restart() { stopPlay(); setHour(1); startPlay(); }

// ---------- head-to-head (tab 4) ----------
function loadRandom(septic) {
  fetch(`/api/compare_random?septic=${septic ? 1 : 0}`).then(r => r.json()).then(p => {
    state.cmp = p;
    $("cmp-pid").textContent = "#" + p.id;
    $("cmp-cat").textContent = p.category || "test set";
    $("cmp-hour-max").textContent = p.max_hour;
    const sl = $("cmp-hour"); sl.max = p.max_hour; sl.value = p.hour;
    renderCompareAt(p.hour);
  });
}

function renderCompareAt(hour) {
  const p = state.cmp; if (!p) return;
  const t = p.trajectory, n = t.iculos.length, idx = Math.max(0, Math.min(hour - 1, n - 1));
  $("cmp-hour-label").textContent = hour;
  const uProb = t.user[idx], tProb = t.tung[idx];
  const ens = Math.round((p.weight_user * uProb + (1 - p.weight_user) * tProb) * 10) / 10;
  setReadout("ro-user", uProb, t.user_alarm[idx]);
  setReadout("ro-tung", tProb, tProb >= p.tung_threshold);
  setReadout("ro-ens", ens, ens >= p.ensemble_threshold);
  const septicNow = t.label[idx] === 1;
  $("ro-truth-state").innerHTML = septicNow
    ? `<span class="tier-CRITICAL">SEPSIS</span>`
    : `<span class="tier-LOW">No sepsis</span>`;
  const op = $("ro-truth-onset");
  op.textContent = p.true_onset_hour ? `onset @ ICU hour ${p.true_onset_hour}` : "never septic";
  op.className = "pred-pill " + (septicNow ? "septic" : "");
  drawChart(idx);
  $("chart-foot").textContent = `ICU hour ${t.iculos[idx]} · ensemble = ${Math.round(p.weight_user * 100)}% my model + ${Math.round((1 - p.weight_user) * 100)}% Tung`
    + (p.true_onset_hour ? ` · true onset at ICU hour ${p.true_onset_hour}` : "");
}

function setReadout(id, prob, alarm) {
  $(id + "-prob").textContent = prob == null ? "—" : prob + "%";
  const pill = $(id + "-alarm");
  pill.textContent = alarm ? "PREDICTS SEPSIS" : "NO ALARM";
  pill.className = "pred-pill " + (alarm ? "alarm" : "clear");
}

// ---------- inline SVG line chart ----------
function drawChart(cursorIdx) {
  const p = state.cmp, t = p.trajectory;
  const W = Math.max(560, t.iculos.length * 11), H = 300;
  const m = { t: 16, r: 16, b: 30, l: 40 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const xs = t.iculos, n = xs.length;
  const maxY = Math.max(10, Math.ceil(Math.max(...t.user, ...t.tung) / 10) * 10);
  const X = i => m.l + (n <= 1 ? 0 : (i / (n - 1)) * iw);
  const Y = v => m.t + ih - (v / maxY) * ih;
  const path = arr => arr.map((v, i) => (i ? "L" : "M") + X(i).toFixed(1) + " " + Y(v).toFixed(1)).join(" ");
  const gy = [];
  for (let g = 0; g <= maxY; g += Math.max(5, Math.round(maxY / 5 / 5) * 5)) gy.push(g);

  let onsetX = null;
  if (p.true_onset_hour) { const oi = xs.findIndex(h => h >= p.true_onset_hour); if (oi >= 0) onsetX = X(oi); }
  const step = Math.max(1, Math.round(n / 8)), xticks = [];
  for (let i = 0; i < n; i += step) xticks.push(i);

  let svg = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="Predicted risk over ICU hours">`;
  gy.forEach(g => { const y = Y(g).toFixed(1); svg += `<line class="grid-line" x1="${m.l}" y1="${y}" x2="${W - m.r}" y2="${y}"/><text x="${m.l - 6}" y="${(+y + 3)}" text-anchor="end">${g}%</text>`; });
  xticks.forEach(i => { const x = X(i).toFixed(1); svg += `<text x="${x}" y="${H - 10}" text-anchor="middle">${xs[i]}</text>`; });
  svg += `<line class="axis-line" x1="${m.l}" y1="${m.t + ih}" x2="${W - m.r}" y2="${m.t + ih}"/>`;
  if (onsetX != null) svg += `<line class="onset-line" x1="${onsetX.toFixed(1)}" y1="${m.t}" x2="${onsetX.toFixed(1)}" y2="${m.t + ih}"/><text x="${onsetX.toFixed(1)}" y="${m.t - 4}" text-anchor="middle" fill="var(--color-critical)" font-weight="700">onset</text>`;
  const cx = X(cursorIdx).toFixed(1);
  svg += `<line class="cursor-line" x1="${cx}" y1="${m.t}" x2="${cx}" y2="${m.t + ih}"/>`;
  svg += `<path class="s-tung" fill="none" stroke-width="2.5" stroke-linejoin="round" d="${path(t.tung)}"/>`;
  svg += `<path class="s-user" fill="none" stroke-width="2.5" stroke-linejoin="round" d="${path(t.user)}"/>`;
  svg += `<circle class="dot-tung" cx="${cx}" cy="${Y(t.tung[cursorIdx]).toFixed(1)}" r="5"/>`;
  svg += `<circle class="dot-user" cx="${cx}" cy="${Y(t.user[cursorIdx]).toFixed(1)}" r="5"/>`;
  const lx = X(n - 1).toFixed(1);
  const uHigher = t.user[n - 1] >= t.tung[n - 1];
  const uY = (Y(t.user[n - 1]) + (uHigher ? -8 : 16)).toFixed(1);
  const tY = (Y(t.tung[n - 1]) + (uHigher ? 16 : -8)).toFixed(1);
  svg += `<text class="lbl-user" x="${lx}" y="${uY}" text-anchor="end">My ${t.user[n - 1]}%</text>`;
  svg += `<text class="lbl-tung" x="${lx}" y="${tY}" text-anchor="end">Tung ${t.tung[n - 1]}%</text>`;
  svg += `</svg>`;
  $("chart").innerHTML = svg;
}
