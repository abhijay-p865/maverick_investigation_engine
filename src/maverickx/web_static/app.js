const fileInput = document.getElementById("file-input");
const dropZone = document.getElementById("drop-zone");
const selectedFile = document.getElementById("selected-file");
const analyzeButton = document.getElementById("analyze-button");
const demoButton = document.getElementById("demo-button");
const loading = document.getElementById("loading");
const errorPanel = document.getElementById("error");
const results = document.getElementById("results");
let currentFile = null;
let currentCharts = null;

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function chooseFile(file) {
  currentFile = file || null;
  selectedFile.textContent = currentFile ? `${currentFile.name} · ${(currentFile.size / 1024 / 1024).toFixed(2)} MB` : "";
  analyzeButton.disabled = !currentFile;
}

fileInput.addEventListener("change", () => chooseFile(fileInput.files[0]));
["dragenter", "dragover"].forEach(type => dropZone.addEventListener(type, event => {
  event.preventDefault();
  dropZone.classList.add("dragging");
}));
["dragleave", "drop"].forEach(type => dropZone.addEventListener(type, event => {
  event.preventDefault();
  dropZone.classList.remove("dragging");
}));
dropZone.addEventListener("drop", event => {
  const file = event.dataTransfer.files[0];
  if (file) chooseFile(file);
});

function metric(label, value) {
  const card = el("div", "metric");
  card.append(el("span", null, label), el("strong", null, value));
  return card;
}

function formatScore(value) {
  if (value !== 0 && Math.abs(value) < 0.0001) return value.toExponential(2);
  return value.toFixed(4);
}

function evidenceBlock(label, content, className = "") {
  const block = el("div", `evidence-block ${className}`.trim());
  block.append(el("span", "evidence-label", label));
  if (typeof content === "string") {
    block.append(el("p", null, content));
  } else {
    block.append(content);
  }
  return block;
}

function priorityTiming(item, events) {
  const eventIds = item.supporting_event_ids || [];
  const byId = new Map(events.map(event => [event.event_id, event]));
  const linked = eventIds.map(id => byId.get(id)).filter(Boolean);
  if (!linked.length) return "No single anomaly event was linked to this batch-level priority.";
  return linked
    .map(event => `Event ${event.event_id}: ${event.start_h.toFixed(1)}–${event.end_h.toFixed(1)} h`)
    .join(" · ");
}

function renderPriorities(items, events) {
  const list = document.getElementById("priority-list");
  list.replaceChildren();
  if (!items.length) {
    list.append(el("p", "lede compact", "No anomaly-driven investigation priority was generated."));
    return;
  }
  items.forEach(item => {
    const row = el("article", "priority-item");
    const rank = el("div", "priority-rank", String(item.rank).padStart(2, "0"));
    const content = el("div");
    content.append(el("h3", null, item.investigation_title));
    const observations = document.createElement("ul");
    observations.className = "observation-list";
    (item.confirmed_observations?.slice(0, 2) || ["No single dominant observation."])
      .forEach(observation => observations.append(el("li", null, observation)));
    content.append(
      evidenceBlock("Confirmed observation", observations, "confirmed"),
      evidenceBlock("When observed", priorityTiming(item, events), "timing"),
      evidenceBlock("Possible explanation", item.possible_explanation),
      evidenceBlock("Suggested check", item.suggested_operator_verification)
    );
    const score = el("div", "score");
    score.append(el("span", null, "PRIORITY SCORE"), el("strong", null, item.priority_score.toFixed(3)));
    row.append(rank, content, score);
    list.append(row);
  });
}

function renderEvents(events) {
  const list = document.getElementById("events-list");
  list.replaceChildren();
  if (!events.length) {
    list.append(el("p", "lede compact", "No grouped anomaly event crossed the saved v2.1 threshold."));
    return;
  }
  events.forEach((event, index) => {
    const details = el("details", "event");
    if (index === 0) details.open = true;
    const summary = el("summary");
    summary.append(
      el("span", "event-number", String(event.event_id).padStart(2, "0")),
      el("span", "event-title", `${event.start_h.toFixed(1)}–${event.end_h.toFixed(1)} hours`),
      el("span", "event-peak", `Peak ${event.maximum_anomaly_score.toFixed(4)}`)
    );
    const body = el("div", "event-body");
    const evidence = el("div");
    evidence.append(el("h4", null, "Confirmed model observation"), el("p", null, event.interpretation));
    const table = el("table", "feature-table");
    const head = document.createElement("thead");
    const headRow = document.createElement("tr");
    ["Feature", "Observed", "SHAP"].forEach(text => headRow.append(el("th", null, text)));
    head.append(headRow); table.append(head);
    const tbody = document.createElement("tbody");
    event.top_features.forEach(feature => {
      const row = document.createElement("tr");
      row.append(
        el("td", null, feature.feature.replaceAll("_", " ")),
        el("td", null, `${Number(feature.observed_feature_value).toPrecision(4)} ${feature.unit}`),
        el("td", feature.mean_shap_contribution > 0 ? "positive" : null, `${feature.mean_shap_contribution >= 0 ? "+" : ""}${feature.mean_shap_contribution.toFixed(4)}`)
      );
      tbody.append(row);
    });
    table.append(tbody); evidence.append(table);

    const checks = el("div");
    checks.append(el("h4", null, "Possible investigations to verify"));
    const ol = document.createElement("ol");
    event.investigations.forEach(item => {
      const li = document.createElement("li");
      const strong = el("strong", null, item.investigation_title);
      li.append(strong, document.createTextNode(` — ${item.suggested_operator_verification}`));
      ol.append(li);
    });
    checks.append(ol, el("h4", null, "Representative sensor trends"));
    const trends = document.createElement("ul");
    event.sensor_readings.forEach(reading => {
      trends.append(el("li", null, `${reading.sensor.replaceAll("_", " ")}: ${reading.trend}, ${Number(reading.start_value).toPrecision(4)} → ${Number(reading.end_value).toPrecision(4)} ${reading.unit}`));
    });
    checks.append(trends);
    body.append(evidence, checks);
    details.append(summary, body);
    list.append(details);
  });
}

function renderDownloads(downloads) {
  const container = document.getElementById("downloads");
  container.replaceChildren();
  const labels = { json: "Download JSON", markdown: "Download report", timeline_csv: "Timeline CSV" };
  Object.entries(downloads).forEach(([key, file]) => {
    const link = el("a", null, labels[key]);
    const raw = atob(file.content_base64);
    const bytes = Uint8Array.from(raw, character => character.charCodeAt(0));
    link.href = URL.createObjectURL(new Blob([bytes], { type: file.media_type }));
    link.download = file.filename;
    container.append(link);
  });
}

function renderResult(data) {
  document.getElementById("result-batch").textContent = data.batch_id;
  document.getElementById("result-summary").textContent = data.summary.plain_language_summary;
  const metrics = document.getElementById("metrics");
  metrics.replaceChildren(
    metric("Anomaly events", String(data.summary.anomaly_event_count)),
    metric("Flagged windows", `${data.summary.flagged_window_count} / ${data.summary.eligible_window_count}`),
    metric("Maximum score", formatScore(data.summary.maximum_anomaly_score)),
    metric("First priority", data.ranked_investigations[0]?.investigation_title || "None")
  );
  renderDownloads(data.downloads);
  renderPriorities(data.ranked_investigations, data.events);
  renderEvents(data.events);
  const limitations = document.getElementById("limitations-list");
  limitations.replaceChildren(...data.limitations.map(item => el("li", null, item)));
  currentCharts = data.charts;
  document.getElementById("chart-image").src = currentCharts.timeline;
  document.querySelectorAll(".chart-tab").forEach(tab => tab.classList.toggle("active", tab.dataset.chart === "timeline"));
  results.classList.remove("hidden");
  results.scrollIntoView({ behavior: "smooth", block: "start" });
}

document.querySelectorAll(".chart-tab").forEach(tab => tab.addEventListener("click", () => {
  if (!currentCharts) return;
  document.querySelectorAll(".chart-tab").forEach(item => item.classList.remove("active"));
  tab.classList.add("active");
  document.getElementById("chart-image").src = currentCharts[tab.dataset.chart];
}));

async function runAnalysis(endpoint, file = null) {
  errorPanel.classList.add("hidden");
  results.classList.add("hidden");
  loading.classList.remove("hidden");
  analyzeButton.disabled = true;
  demoButton.disabled = true;
  try {
    const request = { method: "POST" };
    if (file) {
      const form = new FormData();
      form.append("file", file);
      request.body = form;
    }
    const response = await fetch(endpoint, request);
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "The server could not analyze this CSV.");
    renderResult(data);
  } catch (error) {
    document.getElementById("error-message").textContent = error.message;
    errorPanel.classList.remove("hidden");
    errorPanel.scrollIntoView({ behavior: "smooth", block: "center" });
  } finally {
    loading.classList.add("hidden");
    analyzeButton.disabled = !currentFile;
    demoButton.disabled = false;
  }
}

analyzeButton.addEventListener("click", () => {
  if (currentFile) runAnalysis("/api/analyze", currentFile);
});

demoButton.addEventListener("click", () => runAnalysis("/api/demo"));
