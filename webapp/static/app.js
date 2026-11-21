const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pretty = (d) => (d || "").replaceAll("_", " ");
const isBooks = (d) => /book/i.test(d || "");

async function api(path, options = {}) {
  const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}
const source = () => $("#source").value;
const target = () => $("#target").value;
const k = () => parseInt($("#k").value, 10) || 10;
const loading = (el, msg = "Working…") => (el.innerHTML = `<div class="loading">${esc(msg)}</div>`);
const fail = (el, e) => (el.innerHTML = `<div class="error">${esc(e.message)}</div>`);

/* ---------- cover tiles: a stable colour per title, tinted by domain ---------- */
function hash(s) { let h = 0; for (const c of String(s)) h = (h * 31 + c.charCodeAt(0)) | 0; return Math.abs(h); }
function initials(title) {
  const words = String(title).replace(/[^\p{L}\p{N} ]/gu, " ").split(/\s+/)
    .filter((w) => w && !/^(the|a|an|of|and|vhs|dvd)$/i.test(w));
  return (words.slice(0, 2).map((w) => w[0]).join("") || "?").toUpperCase();
}
function coverStyle(title, domain) {
  const base = isBooks(domain) ? 32 : 245;          // amber for books, indigo for film/TV
  const hue = base + (hash(title) % 40) - 20;
  const light = 38 + (hash(title + "x") % 14);
  return `background: linear-gradient(150deg, hsl(${hue} 62% ${light + 10}%), hsl(${hue + 18} 58% ${light}%))`;
}
const cover = (title, domain, cls = "cover") =>
  `<div class="${cls}" style="${coverStyle(title, domain)}">${cls === "cover" ? esc(initials(title)) : ""}</div>`;

/* ---------- setup ---------- */
async function init() {
  try {
    const [h, stats] = await Promise.all([api("/api/health"), api("/api/domains")]);
    const opts = h.domains.map((d) => `<option value="${d}">${pretty(d)}</option>`).join("");
    $("#source").innerHTML = opts;
    $("#target").innerHTML = opts;
    if (h.domains.length > 1) $("#target").value = h.domains[1];
    $("#statUsers").textContent = h.users.toLocaleString();
    $("#sideStats").insertAdjacentHTML("beforeend", Object.entries(stats).map(([d, s]) =>
      `<div class="stat-row"><span><i class="dot ${isBooks(d) ? "b" : "m"}"></i>${pretty(d)}</span>
       <b>${s.items.toLocaleString()}</b></div>`).join(""));
  } catch (e) {
    fail($("#userOut"), new Error("No data loaded: " + e.message + " — run: python app.py preprocess all"));
  }
}

document.querySelectorAll(".nav-item").forEach((t) =>
  t.addEventListener("click", () => {
    document.querySelectorAll(".nav-item").forEach((x) => x.classList.toggle("active", x === t));
    document.querySelectorAll(".view").forEach((v) => (v.hidden = v.id !== "tab-" + t.dataset.tab));
    window.scrollTo({ top: 0 });
  }));

$("#k").addEventListener("input", () => ($("#kOut").textContent = k()));
$("#swap").addEventListener("click", () => {
  const s = source(); $("#source").value = target(); $("#target").value = s; resetTaste();
});
$("#source").addEventListener("change", resetTaste);

/* ---------- rendering ---------- */
function recCards(data, { flagHits = false } = {}) {
  const max = Math.max(1, ...data.recommendations.map((r) => r.support || 0));
  return `<div class="rec-grid">${data.recommendations.map((r, i) => {
    const hit = flagHits && r.liked_by_user;
    const pct = Math.round(((r.support || 0) / max) * 100);
    return `<article class="rec-card${hit ? " hit" : ""}" style="animation-delay:${i * 30}ms">
      <span class="rank">${i + 1}</span>
      ${cover(r.title, data.target_domain)}
      <div>
        ${hit ? '<span class="hit-badge">✓ they liked it</span>' : ""}
        <div class="rec-title" title="${esc(r.title)}">${esc(r.title)}</div>
        <div class="rec-why">${esc(r.reason)}</div>
      </div>
      <div class="support"><span>support</span><div class="bar"><i style="width:${pct}%"></i></div><span>${r.support}</span></div>
    </article>`;
  }).join("")}</div>`;
}
const chips = (items, domain) =>
  items.map((x) => `<span class="chip ${isBooks(domain) ? "b" : "m"}" title="${esc(x.title)}">${esc(x.title)}</span>`).join("");

/* ---------- discover for a user ---------- */
$("#randomUser").addEventListener("click", async () => {
  try {
    const d = await api(`/api/users?source=${source()}&target=${target()}&limit=1&random=1`);
    $("#userId").value = d.users[0]?.user_id || "";
    recommendUser();
  } catch (e) { fail($("#userOut"), e); }
});
$("#recUser").addEventListener("click", recommendUser);
$("#userId").addEventListener("keydown", (e) => e.key === "Enter" && recommendUser());

async function recommendUser() {
  const out = $("#userOut"); loading(out, "Finding recommendations…");
  const body = { source_domain: source(), target_domain: target(), k: k() };
  const id = $("#userId").value.trim();
  if (id) body.user_id = id;
  try {
    const d = await api("/api/recommend", { method: "POST", body: JSON.stringify(body) });
    $("#userId").value = d.user_id;
    // Honest check: their own data stays hidden, but items they already liked may be predicted.
    const check = await api("/api/recommend", { method: "POST",
      body: JSON.stringify({ ...body, user_id: d.user_id, include_known: true }) });
    const truth = d.already_liked_in_target.length;
    const hits = check.recommendations.filter((r) => r.liked_by_user).length;
    const pct = Math.round((hits / Math.max(1, check.recommendations.length)) * 100);

    out.innerHTML = `
      <div class="profile">
        <div class="profile-top">
          <span class="profile-id">${esc(d.user_id)}</span>
          <span class="profile-meta">liked ${d.seeds.length} in ${pretty(d.source_domain)} · ${truth} in ${pretty(d.target_domain)}</span>
        </div>
        <div class="chips">${chips(d.seeds, d.source_domain)}</div>
      </div>

      <div class="section-title"><h2>New for them in ${pretty(d.target_domain)}</h2>
        <span class="pill">${esc(d.strategy)}</span></div>
      ${recCards(d)}

      <div class="reality">
        <div class="ring" style="--p:${pct}"><span>${hits}/${check.recommendations.length}</span></div>
        <p><b>Reality check.</b> With this user's own ratings hidden, ${hits} of the top ${check.recommendations.length}
        predictions are ${pretty(d.target_domain)} items they really liked (out of ${truth}).
        ${hits ? "The matches are highlighted below." : "No matches for this user — try another one."}</p>
      </div>
      ${hits ? `<div class="section-title"><h2>Predictions vs. what they liked</h2></div>
        ${recCards(check, { flagHits: true })}` : ""}`;
  } catch (e) { fail(out, e); }
}

/* ---------- build your taste ---------- */
const picked = new Map();
let searchTimer;
function resetTaste() {
  picked.clear(); renderPicked();
  $("#searchResults").innerHTML = `<li class="muted">Type at least 2 characters.</li>`;
  $("#q").value = ""; $("#tasteOut").innerHTML = "";
}
$("#q").addEventListener("input", () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(runSearch, 220);
});
async function runSearch() {
  const q = $("#q").value.trim();
  if (q.length < 2) { $("#searchResults").innerHTML = `<li class="muted">Type at least 2 characters.</li>`; return; }
  try {
    const d = await api(`/api/items/search?domain=${source()}&q=${encodeURIComponent(q)}&limit=25`);
    $("#searchResults").innerHTML = d.items.length
      ? d.items.map((i) => `<li data-asin="${esc(i.asin)}" data-title="${esc(i.title)}" class="${picked.has(i.asin) ? "on" : ""}">
          ${cover(i.title, source(), "mini-cover")}<span class="t">${esc(i.title)}</span><small>${i.likes} likes</small></li>`).join("")
      : `<li class="muted">No ${pretty(source())} titles match “${esc(q)}”.</li>`;
  } catch (e) { fail($("#searchResults"), e); }
}
$("#searchResults").addEventListener("click", (e) => {
  const li = e.target.closest("li[data-asin]");
  if (!li) return;
  if (picked.has(li.dataset.asin)) picked.delete(li.dataset.asin);
  else picked.set(li.dataset.asin, li.dataset.title);
  li.classList.toggle("on", picked.has(li.dataset.asin));
  renderPicked();
});
function renderPicked() {
  $("#pickCount").textContent = picked.size;
  $("#picked").innerHTML = picked.size
    ? [...picked].map(([a, t]) => `<div class="picked-item">${cover(t, source(), "mini-cover")}
        <span title="${esc(t)}">${esc(t)}</span><button data-remove="${esc(a)}" aria-label="Remove">×</button></div>`).join("")
    : `<p class="muted">Nothing picked yet.</p>`;
  $("#recTaste").disabled = picked.size === 0;
}
$("#picked").addEventListener("click", (e) => {
  const a = e.target.dataset.remove;
  if (!a) return;
  picked.delete(a); renderPicked();
  document.querySelector(`#searchResults li[data-asin="${CSS.escape(a)}"]`)?.classList.remove("on");
});
$("#recTaste").addEventListener("click", async () => {
  const out = $("#tasteOut"); loading(out, "Finding recommendations…");
  try {
    const d = await api("/api/recommend", { method: "POST", body: JSON.stringify(
      { source_domain: source(), target_domain: target(), k: k(), liked_asins: [...picked.keys()] }) });
    out.innerHTML = `<div class="section-title"><h2>Because you picked ${d.seeds.length} ${pretty(d.source_domain)} title${d.seeds.length > 1 ? "s" : ""}</h2>
      <span class="pill">${esc(d.strategy)}</span></div>${recCards(d)}`;
    out.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (e) { fail(out, e); }
});

/* ---------- evaluate ---------- */
const METRICS = [
  ["hit_rate", "Hit rate", "≥1 liked item in top K"],
  ["precision", "Precision", "share of picks liked"],
  ["recall", "Recall", "share of likes found"],
  ["ndcg", "NDCG", "ranking quality"],
  ["diversity", "Diversity", "how varied the list is"],
];
const lift = (a, b) => (b > 0 ? ((a - b) / b) * 100 : 0);

$("#runEval").addEventListener("click", async () => {
  const out = $("#evalOut"); loading(out, "Evaluating… this takes a few seconds.");
  try {
    const n = $("#nUsers").value || 200;
    const d = await api(`/api/evaluate?source=${source()}&target=${target()}&k=${k()}&n=${n}`);
    const eng = d.methods["co-occurrence"], pop = d.methods["popularity"];
    const kpi = (label, value, sub, cls = "") =>
      `<div class="kpi"><div class="label">${label}</div><div class="value ${cls}">${value}</div><div class="sub">${sub}</div></div>`;
    const hl = lift(eng.hit_rate, pop.hit_rate), nl = lift(eng.ndcg, pop.ndcg);
    const sign = (v) => (v >= 0 ? "+" : "") + v.toFixed(0) + "%";

    const rows = METRICS.map(([key, name, hint]) => {
      const a = eng[key], b = pop[key], max = Math.max(a, b) || 1;
      const better = key !== "diversity" ? a >= b : null;
      return `<div class="metric">
        <div class="name">${name}@${d.k}<small>${hint}</small></div>
        <div class="track engine"><i style="width:${(a / max) * 100}%"></i></div>
        <div class="num ${better === true ? "win" : ""}">${a.toFixed(3)}</div>
        <div class="track pop"><i style="width:${(b / max) * 100}%"></i></div>
        <div class="num ${better === false ? "win" : ""}">${b.toFixed(3)}</div>
      </div>`;
    }).join("");

    out.innerHTML = `
      <div class="eval-hero">
        ${kpi(`Hit rate @${d.k}`, (eng.hit_rate * 100).toFixed(1) + "%", `users with ≥1 real like in their top ${d.k}`)}
        ${kpi("vs. popularity", sign(hl), "relative hit-rate change", hl >= 0 ? "up" : "down")}
        ${kpi("NDCG lift", sign(nl), "ranking quality change", nl >= 0 ? "up" : "down")}
        ${kpi("Users tested", d.users_evaluated.toLocaleString(), `of ${d.users_available.toLocaleString()} · ${pretty(d.source_domain)} → ${pretty(d.target_domain)}`)}
      </div>
      <div class="legend"><span><i style="background:var(--accent)"></i>Co-occurrence engine</span>
        <span><i style="background:var(--muted);opacity:.55"></i>Popularity baseline</span></div>
      <div class="metrics">${rows}</div>`;
  } catch (e) { fail(out, e); }
});

/* ---------- shareable links: #user=<id>, #taste, #evaluate, #api ---------- */
function showTab(name) {
  document.querySelector(`.nav-item[data-tab="${name}"]`)?.click();
}
async function openFromHash() {
  const h = decodeURIComponent(location.hash.slice(1));
  if (h.startsWith("user=")) {
    showTab("user"); $("#userId").value = h.slice(5); recommendUser();
  } else if (h === "evaluate") {
    showTab("evaluate"); $("#runEval").click();
  } else if (["taste", "api"].includes(h)) {
    showTab(h);
  }
}

init().then(openFromHash);
