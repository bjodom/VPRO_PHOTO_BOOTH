"use strict";

const POLL_IDLE_MS = 700;
const POLL_BUSY_MS = 400;
const AUTO_CAPTURE_HOLD_MS = 1500;
const AUTO_CAPTURE_GRACE_MS = 1000;

let current = null;
let timer = null;
let scenesRendered = false;
let pending = false;
let countdownTimer = null;
let countdownSession = null;
let autoCaptureGoodSince = null;
let autoCaptureLastGoodAt = null;
let autoCaptureSession = null;
let autoCaptureSuppressedSession = null;
let connected = false;
let polling = false;

async function api(path, options) {
  const response = await fetch(path, { ...options, signal: AbortSignal.timeout(8000) });
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
  if (pending) return;
  pending = true;
  updateControls();
  document.getElementById("action-error").hidden = true;
  try {
    render(
      await api(`/api/action/${action}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...(payload || {}), session_id: current?.session_id }),
      })
    );
  } catch (err) {
    // A rejected transition usually means the session moved on (timeout); resync rather than guess.
    const notice = document.getElementById("action-error");
    notice.textContent = err.message;
    notice.hidden = false;
  } finally {
    pending = false;
    updateControls();
  }
}

function updateControls() {
  document.querySelectorAll("[data-action], .scene, #capture-btn, #custom-location, #custom-location-submit").forEach(button => {
    button.disabled = pending || !connected || countdownTimer !== null;
  });
  document.getElementById("consent-continue").disabled ||= !document.getElementById("consent-box").checked;
  const busy = !!current?.busy || ["loading", "failed"].includes(current?.status?.renderer_state);
  document.querySelector('[data-action="start"]').disabled ||= busy || current?.status?.camera_open === false;
  document.querySelector('[data-action="accept_capture"]').disabled ||= !!current?.busy;
  document.querySelector('[data-action="previous_camera"]').disabled ||= current?.status?.camera_index <= 0 || !!current?.busy;
  document.querySelector('[data-action="next_camera"]').disabled ||= !!current?.busy;
}

function cancelCountdown({ suppressAuto = false } = {}) {
  if (suppressAuto) autoCaptureSuppressedSession = countdownSession || current?.session_id || null;
  clearInterval(countdownTimer);
  countdownTimer = null;
  countdownSession = null;
  autoCaptureGoodSince = null;
  autoCaptureLastGoodAt = null;
  autoCaptureSession = null;
  document.getElementById("countdown").hidden = true;
  document.getElementById("cancel-countdown").hidden = true;
  updateControls();
}

function beginCountdown() {
  if (!connected || pending || current?.state !== "pose") return;
  countdownSession = current.session_id;
  let remaining = Math.max(0, current.countdown_seconds ?? 3);
  const overlay = document.getElementById("countdown");
  const capture = () => {
    const valid = connected && current?.state === "pose" && current.session_id === countdownSession;
    cancelCountdown();
    if (valid) act("capture");
  };
  if (remaining === 0) { capture(); return; }
  overlay.hidden = false;
  overlay.textContent = remaining;
  document.getElementById("cancel-countdown").hidden = false;
  countdownTimer = setInterval(() => {
    remaining -= 1;
    overlay.textContent = remaining;
    if (remaining <= 0) capture();
  }, 1000);
  updateControls();
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
    button.innerHTML = `<img alt="" loading="lazy"><b></b><span></span>`;
    button.querySelector("img").src = scene.preview_url || `/static/${scene.key}.jpg?v=event-3`;
    button.querySelector("b").textContent = scene.label;
    button.querySelector("span").textContent = scene.country || "";
    button.addEventListener("click", () => act("choose_scene", { scene: scene.key }));
    grid.appendChild(button);
  });
  scenesRendered = true;
}

function render(snapshot) {
  const previousCameraIndex = current?.status?.camera_index;
  const changed = !current || current.state !== snapshot.state ||
    current.session_id !== snapshot.session_id;
  const cameraChanged = previousCameraIndex !== undefined && previousCameraIndex !== snapshot.status?.camera_index;
  current = snapshot;
  if (changed || snapshot.state !== "pose") {
    autoCaptureGoodSince = null;
    autoCaptureLastGoodAt = null;
    autoCaptureSession = null;
    autoCaptureSuppressedSession = null;
  }
  if (countdownTimer !== null && (snapshot.state !== "pose" || snapshot.session_id !== countdownSession)) cancelCountdown();

  renderScenes(snapshot.scenes);
  show(snapshot.state);
  document.getElementById("selected-destination").textContent = snapshot.scene === "custom"
    ? snapshot.custom_location || "" : snapshot.scenes.find(scene => scene.key === snapshot.scene)?.label || "";
  document.getElementById("retention-copy").textContent = snapshot.retention_hours > 0
    ? `Photos and working images are deleted after ${snapshot.retention_hours} hours, except while a session or download link is active.`
    : "Automatic file deletion is disabled. Ask staff about retention before continuing.";
  const labels = {preparing:"Preparing your portrait",composing:"Framing your portrait",generating:"Creating your destination",delivery:"Preparing your download"};
  document.getElementById("generation-stage").textContent = labels[snapshot.stage] || "Creating your portrait";
  const elapsed = Math.floor(snapshot.elapsed_seconds || 0);
  document.getElementById("generation-time").textContent = elapsed > 0
    ? `${elapsed}s elapsed${snapshot.estimated_seconds ? ` · Usually about ${Math.ceil(snapshot.estimated_seconds)}s` : ""}` : "";
  document.querySelectorAll("[data-stage]").forEach(el => el.classList.toggle("current", el.dataset.stage === snapshot.stage));
  document.getElementById("availability").textContent = snapshot.busy ? "Finishing the previous portrait" :
    snapshot.status?.renderer_state === "loading" ? "Preparing the studio" :
    snapshot.status?.renderer_state === "failed" ? "Studio unavailable. Please ask staff." :
    snapshot.status?.camera_open === false ? "Camera unavailable. Please ask staff." : "";
  document.getElementById("result-notice").hidden = !snapshot.degraded;
  document.getElementById("result-notice").textContent = "Your classic portrait is ready. The AI destination could not be completed.";
  document.getElementById("ready-title").textContent = snapshot.degraded ? "Your classic portrait" : "Your photo is ready";

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
  updateAutoCapture(framing);

  if (changed) {
    applyStateAssets(snapshot);
    cancelCountdown();
    document.querySelector(".screen.active h1, .screen.active h2")?.setAttribute("tabindex", "-1");
    document.querySelector(".screen.active h1, .screen.active h2")?.focus({preventScroll:true});
    document.getElementById("screens").scrollTop = 0;
  }
  updateControls();

  const staff = document.getElementById("staff-status");
  if (!document.getElementById("staff-panel").hidden) {
    document.getElementById("staff-camera").textContent = `Camera index: ${snapshot.status?.camera_index ?? "unknown"} · rotate: ${snapshot.status?.camera_rotate ?? "unknown"}°`;
    staff.textContent = JSON.stringify(snapshot.status, null, 2);
  }

  if (cameraChanged && snapshot.state === "pose") {
    const preview = document.getElementById("preview-pose");
    preview.src = `/api/preview.mjpg?t=${Date.now()}`;
  }
}

function updateAutoCapture(framing) {
  if (current?.state !== "pose" || !connected || pending || countdownTimer !== null) return;
  const now = Date.now();
  if (!framing.ok) {
    if (autoCaptureLastGoodAt !== null && now - autoCaptureLastGoodAt <= AUTO_CAPTURE_GRACE_MS) return;
    autoCaptureGoodSince = null;
    autoCaptureLastGoodAt = null;
    autoCaptureSession = null;
    autoCaptureSuppressedSession = null;
    return;
  }
  if (autoCaptureSuppressedSession === current.session_id) return;

  autoCaptureLastGoodAt = now;
  if (autoCaptureSession !== current.session_id) {
    autoCaptureSession = current.session_id;
    autoCaptureGoodSince = now;
    return;
  }
  if (autoCaptureGoodSince !== null && now - autoCaptureGoodSince >= AUTO_CAPTURE_HOLD_MS) {
    beginCountdown();
  }
}

function applyStateAssets(snapshot) {
  if (snapshot.state === "select_scene" || snapshot.state === "idle") {
    document.getElementById("custom-location-form").reset();
  }
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
  }

  if (snapshot.state === "consent") {
    document.getElementById("consent-box").checked = false;
    document.getElementById("consent-continue").disabled = true;
  }
}

async function poll() {
  if (polling) return;
  polling = true;
  try {
    const snapshot = await api("/api/state");
    connected = true;
    document.getElementById("connection").hidden = true;
    if (!pending) render(snapshot);
  } catch (err) {
    connected = false;
    cancelCountdown();
    const notice = document.getElementById("connection");
    notice.textContent = "Connection lost. Reconnecting to the photo booth...";
    notice.hidden = false;
    updateControls();
  } finally {
    polling = false;
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
  updateControls();
});
document.getElementById("capture-btn").addEventListener("click", beginCountdown);
document.getElementById("custom-location-form").addEventListener("submit", event => {
  event.preventDefault();
  act("choose_scene", { scene: "custom", custom_location: document.getElementById("custom-location").value });
});
document.getElementById("cancel-countdown").addEventListener("click", () => cancelCountdown({ suppressAuto: true }));

let staffTimer = null;
const openStaff = () => { document.getElementById("staff-panel").hidden = false; document.getElementById("staff-close").focus(); };
document.getElementById("staff").addEventListener("pointerdown", () => { staffTimer = setTimeout(openStaff, 1800); });
["pointerup", "pointerleave", "pointercancel"].forEach(name => document.getElementById("staff").addEventListener(name, () => clearTimeout(staffTimer)));
document.getElementById("staff").addEventListener("keydown", event => {
  if (event.key === "Enter" && event.ctrlKey) openStaff();
});
document.getElementById("force-reset").addEventListener("click", () => {
  if (confirm("Discard this guest session?")) act("reset");
});
document.getElementById("staff-close").addEventListener("click", () => {
  document.getElementById("staff-panel").hidden = true;
  document.getElementById("staff").focus();
});

document.addEventListener("keydown", event => {
  if (event.key === "Escape") { cancelCountdown(); document.getElementById("staff-panel").hidden = true; }
});

poll();
