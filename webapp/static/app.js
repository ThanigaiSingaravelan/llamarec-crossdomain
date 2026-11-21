const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pretty = (d) => d.replaceAll("_", " ");

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" }, ...options,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}
const source = () => $("#source").value;
const target = () => $("#target").value;
const k = () => Math.max(1, Math.min(50, parseInt($("#k").value, 10) || 10));
const busy = (el, msg = "Working…") => (el.innerHTML = `<p class="meta">${msg}</p>`);
const fail = (el, e) => (el.innerHTML = `<p class="err">${esc(e.message)}</p>`);

/* ---------- setup ---------- */
async function init() {
  try {
    const h = await api("/api/health");
    const opts = h.domains.map((d) => `<option value="${d}">${pretty(d)}</option>`).join("");
    $("#source").innerHTML = opts;
    $("#target").innerHTML = opts;
    if (h.domains.length > 1) $("#target").value = h.domains[1];
    $("#status").textContent = `${h.users.toLocaleString()} users · ${h.domains.map(pretty).join(", ")}`;
    $("#status").className = "status ok";
  } catch (e) {
    $("#status").textContent = "No data loaded: " + e.message + " (run: python app.py preprocess all)";
    $("#status").className = "status err";
  }
}

document.querySelectorAll(".tab").forEach((t) =>
  t.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === t));
    document.querySelectorAll(".panel").forEach((p) => (p.hidden = p.id !== "tab-" + t.dataset.tab));
  }));

$("#swap").addEventListener("click", () => {
  const s = source(); $("#source").value = target(); $("#target").value = s;
  picked.clear(); renderPicked(); $("#searchResults").innerHTML = "";
});
$("#source").addEventListener("change", () => { picked.clear(); renderPicked(); });

/* ---------- rendering ---------- */
function renderRecs(data, truth = new Set()) {
  const items = data.recommendations.map((r, i) => {
    const hit = truth.has(r.asin);
    return `<li class="rec${hit ? " hit" : ""}"><span class="n">${i + 1}</span><div>
      <div class="t">${esc(r.title)}${hit ? '<span class="badge">they liked this</span>' : ""}</div>
      <div class="why">${esc(r.reason)}</div></div></li>`;
  }).join("");
  return `<h3>Recommended ${pretty(data.target_domain)}</h3>
    <p class="meta">Strategy: ${esc(data.strategy)}</p><ol class="recs">${items}</ol>`;
}
const list = (items) => items.map((x) => `<span class="chip">${esc(x.title)}</span>`).join("");

/* ---------- for a user ---------- */
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
  const out = $("#userOut"); busy(out);
  const body = { source_domain: source(), target_domain: target(), k: k() };
  const id = $("#userId").value.trim();
  if (id) body.user_id = id;
  try {
    const d = await api("/api/recommend", { method: "POST", body: JSON.stringify(body) });
    $("#userId").value = d.user_id;
    const truth = new Set(d.already_liked_in_target.map((x) => x.asin));
    // Honest check: same user (their own data still hidden), but items they already liked are allowed.
    const check = await api("/api/recommend", { method: "POST", body: JSON.stringify(
      { ...body, user_id: d.user_id, include_known: true }) });
    const hits = check.recommendations.filter((r) => r.liked_by_user).length;
    out.innerHTML = `<h3>${esc(d.user_id)} liked in ${pretty(d.source_domain)} (${d.seeds.length})</h3>
      <div class="chips">${list(d.seeds.slice(0, 15))}${d.seeds.length > 15 ? `<span class="meta">+${d.seeds.length - 15} more</span>` : ""}</div>
      ${renderRecs(d)}
      <h3>Reality check</h3>
      <p class="meta">This user really liked ${truth.size} ${pretty(d.target_domain)} item(s). With their own ratings
      hidden from the model, ${hits} of the top ${k()} predictions are items they actually liked (highlighted).</p>
      ${renderRecs(check, truth)}`;
  } catch (e) { fail(out, e); }
}

/* ---------- pick what you like ---------- */
const picked = new Map();
let searchTimer;
$("#q").addEventListener("input", () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(async () => {
    const q = $("#q").value.trim();
    if (q.length < 2) { $("#searchResults").innerHTML = ""; return; }
    try {
      const d = await api(`/api/items/search?domain=${source()}&q=${encodeURIComponent(q)}&limit=15`);
      $("#searchResults").innerHTML = d.items.length
        ? d.items.map((i) => `<li data-asin="${esc(i.asin)}" data-title="${esc(i.title)}">
            <span>${esc(i.title)}</span><small>${i.likes} likes</small></li>`).join("")
        : `<li class="meta">No matches</li>`;
    } catch (e) { fail($("#searchResults"), e); }
  }, 250);
});
$("#searchResults").addEventListener("click", (e) => {
  const li = e.target.closest("li[data-asin]");
  if (!li) return;
  picked.set(li.dataset.asin, li.dataset.title); renderPicked();
});
function renderPicked() {
  $("#picked").innerHTML = [...picked].map(([a, t]) =>
    `<span class="chip">${esc(t)}<button data-remove="${esc(a)}" aria-label="Remove">×</button></span>`).join("");
  $("#recTaste").disabled = picked.size === 0;
}
$("#picked").addEventListener("click", (e) => {
  const a = e.target.dataset.remove; if (a) { picked.delete(a); renderPicked(); }
});
$("#recTaste").addEventListener("click", async () => {
  const out = $("#tasteOut"); busy(out);
  try {
    const d = await api("/api/recommend", { method: "POST", body: JSON.stringify(
      { source_domain: source(), target_domain: target(), k: k(), liked_asins: [...picked.keys()] }) });
    out.innerHTML = renderRecs(d);
  } catch (e) { fail(out, e); }
});

/* ---------- evaluate ---------- */
$("#runEval").addEventListener("click", async () => {
  const out = $("#evalOut"); busy(out, "Evaluating… this can take a few seconds.");
  try {
    const n = $("#nUsers").value || 200;
    const d = await api(`/api/evaluate?source=${source()}&target=${target()}&k=${k()}&n=${n}`);
    const methods = Object.keys(d.methods);
    const metrics = ["hit_rate", "precision", "recall", "ndcg", "diversity"];
    const rows = metrics.map((m) => {
      const vals = methods.map((x) => d.methods[x][m]);
      const best = Math.max(...vals);
      return `<tr><td>${m.replace("_", " ")}@${d.k}</td>${vals.map((v) =>
        `<td class="num${v === best && m !== "diversity" ? " win" : ""}">${v.toFixed(4)}</td>`).join("")}</tr>`;
    }).join("");
    out.innerHTML = `<p class="meta">${pretty(d.source_domain)} → ${pretty(d.target_domain)} ·
      ${d.users_evaluated} of ${d.users_available} users · ${esc(d.note)}</p>
      <div class="table-wrap"><table><tr><th>Metric</th>${methods.map((m) => `<th class="num">${m}</th>`).join("")}</tr>
      ${rows}</table></div>`;
  } catch (e) { fail(out, e); }
});

init();
