"use strict";

const POLL_IDLE_MS = 700;
const POLL_BUSY_MS = 400;

let current = null;
let timer = null;
let scenesRendered = false;

async function api(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      detail = (await response.json()).detail || detail;
    } catch (_) {}
    throw new Error(detail);
  }
  return response.json();
}

async function act(action, payload) {
  try {
    render(
      await api(`/api/action/${action}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload || {}),
      })
    );
  } catch (err) {
    // A rejected transition usually means the session moved on (timeout); resync rather than guess.
    console.warn(`action ${action} failed: ${err.message}`);
    poll();
  }
}

function show(state) {
  document.querySelectorAll(".screen").forEach((el) => {
    el.classList.toggle("active", el.dataset.state === state);
  });
}

function renderScenes(scenes) {
  if (scenesRendered) return;
  const grid = document.getElementById("scene-grid");
  grid.innerHTML = "";
  scenes.forEach((scene) => {
    const button = document.createElement("button");
    button.className = "scene ghost";
    button.innerHTML = `<b></b><span></span>`;
    button.querySelector("b").textContent = scene.label;
    button.querySelector("span").textContent = scene.description;
    button.addEventListener("click", () => act("choose_scene", { scene: scene.key }));
    grid.appendChild(button);
  });
  scenesRendered = true;
}

function render(snapshot) {
  const changed = !current || current.state !== snapshot.state ||
    current.session_id !== snapshot.session_id;
  current = snapshot;

  renderScenes(snapshot.scenes);
  show(snapshot.state);

  const remaining = snapshot.seconds_remaining;
  document.getElementById("timer").textContent =
    remaining === null || remaining === undefined || snapshot.state === "idle"
      ? ""
      : `${Math.ceil(remaining)}s`;

  document.querySelectorAll("[data-error]").forEach((el) => {
    el.textContent = snapshot.error || "Something went wrong.";
  });

  const framing = snapshot.framing || { ok: true, message: "" };
  const hint = document.getElementById("framing-hint");
  hint.textContent = framing.message;
  hint.classList.toggle("good", !!framing.ok);
  // Advisory, not a gate: a guest who cannot reach a green box must still be able to continue.
  document.getElementById("capture-btn").textContent = framing.ok
    ? "Take Photo"
    : "Take Photo Anyway";

  if (changed) {
    applyStateAssets(snapshot);
  }

  const staff = document.getElementById("staff-status");
  if (!document.getElementById("staff-panel").hidden) {
    staff.textContent = JSON.stringify(snapshot.status, null, 2);
  }
}

function applyStateAssets(snapshot) {
  const preview = document.getElementById("preview-pose");
  // Only attach the MJPEG stream while it is visible; it is an open connection.
  if (snapshot.state === "pose") {
    if (!preview.src) preview.src = `/api/preview.mjpg?t=${Date.now()}`;
  } else {
    preview.removeAttribute("src");
  }

  if (snapshot.state === "review") {
    document.getElementById("review-img").src = `/api/capture.jpg?t=${Date.now()}`;
  }

  if (snapshot.state === "ready") {
    document.getElementById("final-img").src = `/api/final.jpg?t=${Date.now()}`;
    document.getElementById("qr").innerHTML = snapshot.qr_svg || "";
    document.getElementById("ready-url").textContent = snapshot.delivery_url || "";
    const caption = document.getElementById("caption");
    caption.textContent = snapshot.caption || "";
    caption.hidden = true;
    document.getElementById("caption-btn").textContent = "Show caption";
  }

  if (snapshot.state === "consent") {
    document.getElementById("consent-box").checked = false;
    document.getElementById("consent-continue").disabled = true;
  }
}

async function poll() {
  try {
    render(await api("/api/state"));
  } catch (err) {
    console.warn(`state poll failed: ${err.message}`);
  }
  const busy = current && (current.state === "generating" || current.state === "pose");
  clearTimeout(timer);
  timer = setTimeout(poll, busy ? POLL_BUSY_MS : POLL_IDLE_MS);
}

document.addEventListener("click", (event) => {
  const target = event.target.closest("[data-action]");
  if (target) act(target.dataset.action);
});

document.getElementById("consent-box").addEventListener("change", (event) => {
  document.getElementById("consent-continue").disabled = !event.target.checked;
});

document.getElementById("caption-btn").addEventListener("click", () => {
  const caption = document.getElementById("caption");
  caption.hidden = !caption.hidden;
  document.getElementById("caption-btn").textContent = caption.hidden
    ? "Show caption"
    : "Hide caption";
});

document.getElementById("staff").addEventListener("click", () => {
  const panel = document.getElementById("staff-panel");
  panel.hidden = !panel.hidden;
});
document.getElementById("staff-close").addEventListener("click", () => {
  document.getElementById("staff-panel").hidden = true;
});

document.addEventListener("contextmenu", (event) => event.preventDefault());

poll();
