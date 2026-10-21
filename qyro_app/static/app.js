"use strict";

const $ = (sel) => document.querySelector(sel);
const state = { file: null, image: null, result: null, answers: {} };

const RECOMMENDATIONS = {
  routine_dermatologist_review: { cls: "rec-routine", title: "Routine dermatologist review", text: "No red flags found. A dermatologist should still review these findings." },
  priority_dermatologist_review: { cls: "rec-priority", title: "Dermatologist review advised soon", text: "Some findings or answers suggest this should be reviewed with priority." },
  seek_prompt_medical_care: { cls: "rec-urgent", title: "Seek prompt medical care", text: "Reported symptoms need in-person medical assessment. Do not wait for this report." },
  retake_photo: { cls: "rec-retake", title: "Please retake the photo", text: "The photo is not clear enough to analyze. Use even lighting, hold still and fill the frame with the face." },
};

const pretty = (s) => (s || "").replace(/_/g, " ");
const pct = (p) => `${Math.round(p * 100)}%`;
const el = (tag, attrs = {}, ...children) => {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else node.setAttribute(k, v);
  }
  for (const c of children) node.append(c instanceof Node ? c : document.createTextNode(c ?? ""));
  return node;
};

// Tri-state yes / no / unsure answers, so "unknown" is never turned into "no".
for (const row of document.querySelectorAll(".tri")) {
  const field = row.dataset.field;
  const seg = el("span", { class: "seg", role: "group", "aria-label": row.textContent.trim() });
  for (const [label, value] of [["Yes", true], ["No", false], ["Not sure", null]]) {
    const b = el("button", { type: "button", "aria-pressed": String(value === null) }, label);
    b.addEventListener("click", () => {
      state.answers[field] = value;
      seg.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    });
    seg.append(b);
  }
  row.append(seg);
}

function setFile(file) {
  if (!file) return;
  state.file = file;
  $("#file-name").hidden = false;
  const size = file.size < 1024 * 1024 ? `${Math.max(1, Math.round(file.size / 1024))} KB` : `${(file.size / 1024 / 1024).toFixed(1)} MB`;
  $("#file-name").textContent = `Selected: ${file.name} (${size})`;
  $("#analyze").disabled = false;
  const img = new Image();
  img.onload = () => { state.image = img; };
  img.src = URL.createObjectURL(file);
}

$("#photo").addEventListener("change", (e) => setFile(e.target.files[0]));
const dz = $("#dropzone");
["dragenter", "dragover"].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
["dragleave", "drop"].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
dz.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));

function questionnaire() {
  const form = new FormData($("#questionnaire"));
  const q = { duration: form.get("duration") || "unknown" };
  const age = form.get("age_years");
  if (age !== "" && age !== null) q.age_years = Number(age);
  const treatment = (form.get("current_treatment") || "").trim();
  if (treatment) q.current_treatment = treatment;
  for (const [k, v] of Object.entries(state.answers)) if (v !== null) q[k] = v;
  return q;
}

function show(which) {
  for (const id of ["empty", "loading", "output"]) $("#" + id).hidden = id !== which;
}

$("#questionnaire").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!state.file) return;
  $("#form-error").hidden = true;
  $("#analyze").disabled = true;
  show("loading");
  const body = new FormData();
  body.append("image", state.file);
  body.append("questionnaire", JSON.stringify(questionnaire()));
  try {
    const res = await fetch("/api/v1/analyze", { method: "POST", body });
    const data = await res.json();
    if (!res.ok) throw new Error(data?.detail?.message || data?.detail?.code || `Request failed (${res.status})`);
    state.result = data;
    render(data);
  } catch (err) {
    show("empty");
    $("#form-error").hidden = false;
    $("#form-error").textContent = err.message;
  } finally {
    $("#analyze").disabled = false;
  }
});

function drawImage() {
  const canvas = $("#canvas");
  const img = state.image;
  if (!img) return;
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0);
  const boxes = state.result?.vision_findings?.detection?.boxes || [];
  if (!$("#show-boxes").checked) return;
  const lw = Math.max(2, Math.round(Math.min(img.naturalWidth, img.naturalHeight) / 300));
  ctx.lineWidth = lw;
  ctx.font = `${lw * 6}px sans-serif`;
  for (const b of boxes) {
    ctx.strokeStyle = b.confidence >= 0.5 ? "#ff3b6b" : "#ffb020";
    ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
    ctx.fillStyle = ctx.strokeStyle;
    ctx.fillText(pct(b.confidence), b.x1, Math.max(lw * 6, b.y1 - lw));
  }
}
$("#show-boxes").addEventListener("change", drawImage);

function classificationCard(label, c) {
  if (!c.model.available) {
    return el("div", { class: "card unavailable" },
      el("div", { class: "label" }, label),
      el("div", { class: "value" }, "Not available"),
      el("div", { class: "sub" }, c.model.note || "Model not loaded."));
  }
  const bars = el("ul", { class: "bars" });
  for (const p of c.probabilities.slice(0, 4)) {
    bars.append(el("li", {},
      el("span", { class: "name", title: pretty(p.label) }, pretty(p.label)),
      el("span", { class: "track" }, el("span", { class: "fill", style: `width:${(p.probability * 100).toFixed(1)}%` })),
      el("span", { class: "pct" }, pct(p.probability))));
  }
  return el("div", { class: "card" },
    el("div", { class: "label" }, label),
    el("div", { class: "value" }, pretty(c.top_label)),
    el("div", { class: "sub" }, `Model confidence ${pct(c.top_probability)}`),
    bars);
}

function renderExplanation(r) {
  const x = r.explanation;
  $("#explanation-section").hidden = !x;
  if (!x) return;
  const badge = $("#explain-badge");
  if (x.source === "llm") {
    badge.className = "tag";
    badge.textContent = `Written by ${x.provider} (${x.model}) · passed safety check`;
  } else {
    badge.className = "tag template";
    badge.textContent = "Standard text · AI explanation off";
    badge.title = x.fallback_reason || "";
  }
  $("#explain-text").textContent = x.explanation;
  $("#explain-steps").replaceChildren(...x.next_steps.map((s) =>
    el("li", {}, s.text, ...s.source_ids.map((id) => el("a", { class: "cite", href: `#src-${id}` }, id)))));
  $("#explain-uncertainty").textContent = x.uncertainty_note;
  $("#source-count").textContent = String(r.retrieved_evidence.length);
  $("#sources").replaceChildren(...r.retrieved_evidence.map((e) =>
    el("li", { id: `src-${e.id}` }, el("b", {}, `${e.id} · ${e.title}`), " · ",
      el("a", { href: e.url, target: "_blank", rel: "noopener noreferrer" }, e.source))));
  // Open the sources list when a citation is clicked
  $("#explain-steps").querySelectorAll(".cite").forEach((a) => a.addEventListener("click", () => { $(".sources").open = true; }));
}

function render(r) {
  show("output");
  const rec = RECOMMENDATIONS[r.recommendation];
  $("#recommendation").className = `recommendation ${rec.cls}`;
  $("#recommendation").replaceChildren(el("b", {}, rec.title), el("p", {}, rec.text), el("p", {}, r.summary));

  const mock = r.safety_flags.some((f) => f.code === "mock_mode");
  $("#mode-badge").hidden = !mock;
  $("#mode-badge").textContent = "MOCK MODE · fake model outputs";

  $("#quality").replaceChildren(...r.image_quality.checks.map((c) =>
    el("li", { class: `chip ${c.status}`, title: c.message }, `${pretty(c.name)}: ${c.status}`)));

  const cards = [];
  const v = r.vision_findings;
  if (v) {
    const d = v.detection;
    cards.push(d.model.available
      ? el("div", { class: "card" },
          el("div", { class: "label" }, "Lesions detected"),
          el("div", { class: "value" }, String(d.lesion_count)),
          el("div", { class: "sub" }, d.mean_confidence != null ? `Average confidence ${pct(d.mean_confidence)}` : "No lesions found"))
      : el("div", { class: "card unavailable" }, el("div", { class: "label" }, "Lesions detected"), el("div", { class: "value" }, "Not available")));
    cards.push(classificationCard("Acne type", v.subtype));
    cards.push(classificationCard("Severity", v.severity));
  }
  $("#findings").replaceChildren(...cards);
  $(".image-wrap").hidden = !v;

  renderExplanation(r);

  const order = { urgent: 0, caution: 1, info: 2 };
  $("#flags").replaceChildren(...[...r.safety_flags].sort((a, b) => order[a.level] - order[b.level])
    .map((f) => el("li", { class: f.level }, f.message)));

  const meta = [["Request", r.request_id], ["Schema", r.schema_version]];
  if (v) for (const [k, m] of [["Detector", v.detection.model], ["Subtype", v.subtype.model], ["Severity", v.severity.model]])
    meta.push([k, `${m.name} · ${m.version}${m.available ? "" : " (not available)"}`]);
  if (r.explanation) {
    const x = r.explanation;
    meta.push(["Explanation", x.source === "llm" ? `${x.provider} · ${x.model}` : `template (${x.fallback_reason || "LLM off"})`]);
    if (x.input_tokens != null) meta.push(["LLM tokens", `${x.input_tokens} in / ${x.output_tokens} out`]);
    if (x.validation_issues.length) meta.push(["Safety check", `rejected LLM output: ${x.validation_issues.join("; ")}`]);
  }
  for (const [k, ms] of Object.entries(r.latency_ms)) meta.push([pretty(k), `${ms} ms`]);
  $("#meta").replaceChildren(...meta.flatMap(([k, val]) => [el("dt", {}, k), el("dd", {}, val)]));
  $("#raw").textContent = JSON.stringify(r, null, 2);
  $("#disclaimer").textContent = r.disclaimer;

  if (state.image?.complete) drawImage(); else if (state.image) state.image.onload = drawImage;
}
