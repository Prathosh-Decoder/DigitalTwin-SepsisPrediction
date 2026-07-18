const $ = (id) => document.getElementById(id);
const state = { hour: 24, selected: null, playing: true, timer: null, speed: 1000, request: 0, loading: false };
const order = { ACTIVE: 0, FORECAST: 1, WATCH: 2, STABLE: 3 };

function percent(value) {
  return value == null ? "--" : `${(value * 100).toFixed(1)}%`;
}

function valueText(item) {
  if (item.value == null) return "--";
  const digits = ["HR", "MAP", "O2Sat", "Resp"].includes(item.key) ? 0 : 1;
  return `${item.value.toFixed(digits)} ${item.unit}`;
}

function trendText(trend) {
  return { rising: "↑ Rising", falling: "↓ Falling", steady: "→ Steady" }[trend] || "--";
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;"
  })[char]);
}

async function getJSON(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

function setConnection(ok, label) {
  $("service-status").classList.toggle("degraded", !ok);
  $("service-status").querySelector("span").textContent = label;
}

async function checkHealth() {
  try {
    const health = await getJSON("/api/health");
    setConnection(health.status === "ok", health.status === "ok" ? "Models online" : "Forecast unavailable");
  } catch (_) {
    setConnection(false, "API offline");
  }
}

function renderBed(patient) {
  const selected = state.selected === patient.id ? " selected" : "";
  const vitals = Object.fromEntries(patient.vitals.map((item) => [item.key, item]));
  const active = patient.active_alert.alert ? "ALERT" : "Clear";
  const forecast = patient.forecast.alert ? "ALERT" : percent(patient.forecast.probability);
  return `<button class="bed state-${patient.state.toLowerCase()}${selected}" data-pid="${patient.id}">
    <span class="bed-top"><b>${escapeHtml(patient.bed)}</b><i>${patient.state}</i></span>
    <span class="patient-id">Patient ${String(patient.id).padStart(6, "0")}</span>
    <span class="bed-models">
      <span><small>Active</small><strong>${active}</strong></span>
      <span><small>6h forecast</small><strong>${forecast}</strong></span>
    </span>
    <span class="mini-vitals">
      <span><small>HR</small>${vitals.HR ? valueText(vitals.HR).split(" ")[0] : "--"}</span>
      <span><small>MAP</small>${vitals.MAP ? valueText(vitals.MAP).split(" ")[0] : "--"}</span>
      <span><small>SpO2</small>${vitals.O2Sat ? valueText(vitals.O2Sat).split(" ")[0] : "--"}</span>
    </span>
  </button>`;
}

async function loadWard() {
  if (state.loading) return;
  state.loading = true;
  const requestId = ++state.request;
  try {
    const payload = await getJSON(`/api/twin/beds?hour=${state.hour}`);
    if (requestId !== state.request) return;
    for (const key of ["ACTIVE", "FORECAST", "WATCH", "STABLE"]) {
      $(`count-${key.toLowerCase()}`).textContent = payload.counts[key] || 0;
    }
    $("beds").innerHTML = payload.patients.sort((a, b) => order[a.state] - order[b.state]).map(renderBed).join("");
    document.querySelectorAll(".bed").forEach((button) => button.addEventListener("click", () => selectPatient(+button.dataset.pid)));
    $("updated-at").textContent = `Updated ${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}`;
    setConnection(payload.patients.some((p) => p.forecast.available), "Models online");
    if (state.selected) await loadDetail(state.selected);
  } catch (error) {
    $("beds").innerHTML = `<div class="error-state"><strong>Patient stream unavailable</strong><span>${escapeHtml(error.message)}</span></div>`;
    setConnection(false, "API offline");
  } finally {
    state.loading = false;
  }
}

function selectPatient(patientId) {
  state.selected = patientId;
  document.querySelectorAll(".bed").forEach((bed) => bed.classList.toggle("selected", +bed.dataset.pid === patientId));
  loadDetail(patientId);
}

function setModelReadouts(patient) {
  const forecast = patient.forecast;
  const active = patient.active_alert;
  $("forecast-prob").textContent = percent(forecast.probability);
  $("forecast-trend").textContent = trendText(forecast.trend);
  $("forecast-fill").style.width = `${Math.min(100, (forecast.probability || 0) * 100)}%`;
  $("forecast-threshold").style.left = `${Math.min(100, (forecast.threshold || 0) * 100)}%`;
  $("forecast-threshold").title = `Alert threshold ${percent(forecast.threshold)}`;
  $("forecast-status").textContent = forecast.available ? (forecast.alert ? "Forecast threshold crossed" : `Below ${percent(forecast.threshold)} threshold`) : "Forecast service unavailable";
  $("forecast-status").className = forecast.alert ? "alert-text" : "clear-text";

  $("active-prob").textContent = percent(active.probability);
  $("active-tier").textContent = `${active.tier} · ${trendText(active.trend)}`;
  $("active-criticality").textContent = active.criticality == null ? "--" : `${active.criticality.toFixed(1)} / 100`;
  $("active-status").textContent = active.alert ? "Active alert raised" : "No active alert";
  $("active-status").className = active.alert ? "alert-text" : "clear-text";
}

function drawTrajectory(trajectory) {
  const values = trajectory?.probabilities || [];
  const hours = trajectory?.hours || [];
  const thresholds = trajectory?.thresholds || [];
  if (!values.length) {
    $("trajectory").innerHTML = `<span class="no-data">No trajectory available</span>`;
    return;
  }
  const width = 680, height = 180, left = 40, right = 12, top = 12, bottom = 28;
  const innerW = width - left - right, innerH = height - top - bottom;
  const maxY = Math.max(0.1, ...values, ...thresholds) * 1.12;
  const x = (i) => left + (values.length === 1 ? 0 : (i / (values.length - 1)) * innerW);
  const y = (v) => top + innerH - (v / maxY) * innerH;
  const line = values.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  const threshold = thresholds.length ? thresholds.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ") : "";
  const area = `${line} L${x(values.length - 1)},${top + innerH} L${left},${top + innerH} Z`;
  const ticks = [0, 0.5, 1].map((fraction) => {
    const value = maxY * fraction;
    const py = y(value);
    return `<line x1="${left}" y1="${py}" x2="${width - right}" y2="${py}" class="gridline"/><text x="${left - 7}" y="${py + 4}" class="axis-label" text-anchor="end">${(value * 100).toFixed(0)}%</text>`;
  }).join("");
  $("trajectory").innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Six-hour sepsis forecast trajectory">
    ${ticks}<path d="${area}" class="chart-area"/><path d="${line}" class="chart-line"/>
    ${threshold ? `<path d="${threshold}" class="threshold-line"/>` : ""}
    <circle cx="${x(values.length - 1)}" cy="${y(values.at(-1))}" r="4" class="chart-point"/>
    <text x="${left}" y="${height - 6}" class="axis-label">Hour ${hours[0]}</text>
    <text x="${width - right}" y="${height - 6}" class="axis-label" text-anchor="end">Hour ${hours.at(-1)}</text>
  </svg>`;
}

function renderDrivers(drivers) {
  if (!drivers?.length) {
    $("drivers").innerHTML = `<span class="no-data">No SHAP explanation available</span>`;
    return;
  }
  const max = Math.max(...drivers.map((driver) => driver.impact || 0), 0.000001);
  $("drivers").innerHTML = drivers.map((driver) => {
    const width = Math.max(4, ((driver.impact || 0) / max) * 100);
    const direction = driver.direction === "up" ? "Raises forecast" : "Lowers forecast";
    const featureValue = driver.value == null ? "missing" : Number(driver.value).toFixed(2);
    return `<div class="driver ${driver.direction}">
      <div><span>${escapeHtml(driver.label)}</span><small>${direction} · value ${featureValue}</small></div>
      <div class="driver-track"><i style="width:${width}%"></i></div>
    </div>`;
  }).join("");
}

function renderVitals(vitals) {
  $("vitals").innerHTML = vitals.map((item) => {
    const delta = item.delta == null ? "No prior value" : `${item.delta > 0 ? "+" : ""}${item.delta.toFixed(1)} since last measurement`;
    return `<div class="vital-row ${item.abnormal ? "abnormal" : ""}">
      <span>${escapeHtml(item.label)}</span><strong>${valueText(item)}</strong><small>${escapeHtml(delta)}</small>
    </div>`;
  }).join("");
}

async function loadDetail(patientId) {
  const requestedHour = state.hour;
  try {
    const patient = await getJSON(`/api/twin/beds/${patientId}?hour=${requestedHour}`);
    if (state.selected !== patientId || requestedHour !== state.hour) return;
    $("detail-empty").hidden = true;
    $("detail-content").hidden = false;
    $("detail-bed").textContent = `${patient.bed} · ICU hour ${patient.hour}`;
    $("detail-patient").textContent = `Patient ${String(patient.id).padStart(6, "0")}`;
    $("detail-state").textContent = patient.state;
    $("detail-state").className = `state-pill state-${patient.state.toLowerCase()}`;
    setModelReadouts(patient);
    drawTrajectory(patient.forecast.trajectory);
    renderDrivers(patient.forecast.drivers);
    renderVitals(patient.vitals);
    $("measurement-hour").textContent = `ICU hour ${patient.hour}`;
    $("observation").textContent = patient.narrative.observation;
    $("recommendation").textContent = patient.narrative.recommendation;
    $("narrative-source").textContent = patient.narrative.source.startsWith("openai") ? "LLM" : "Rules fallback";
  } catch (error) {
    $("detail-empty").hidden = false;
    $("detail-content").hidden = true;
    $("detail-empty").innerHTML = `<strong>Patient detail unavailable</strong><span>${escapeHtml(error.message)}</span>`;
  }
}

function setHour(hour) {
  state.hour = Math.max(1, Math.min(72, hour));
  $("hour").value = state.hour;
  $("hour-label").textContent = state.hour;
  $("ward-hour").textContent = `Hour ${state.hour}`;
  loadWard();
}

function startTimer() {
  clearInterval(state.timer);
  state.timer = setInterval(() => setHour(state.hour >= 72 ? 1 : state.hour + 1), state.speed);
}

function togglePlayback() {
  state.playing = !state.playing;
  $("play").textContent = state.playing ? "Pause" : "Play";
  if (state.playing) startTimer(); else clearInterval(state.timer);
}

$("play").addEventListener("click", togglePlayback);
$("restart").addEventListener("click", () => setHour(1));
$("speed").addEventListener("change", (event) => { state.speed = +event.target.value; if (state.playing) startTimer(); });
$("hour").addEventListener("input", (event) => setHour(+event.target.value));

checkHealth();
setHour(state.hour);
startTimer();
