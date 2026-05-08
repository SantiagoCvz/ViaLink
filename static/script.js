const COLORS = {
  car: "var(--car)",
  ambulance: "var(--ambulance)",
  truck: "var(--truck)",
  bus: "var(--bus)",
  motorcycle: "var(--motorcycle)",
};
const ICONS = {
  car: "🚗",
  ambulance: "🚑",
  truck: "🚛",
  bus: "🚌",
  motorcycle: "🏍️",
};

let streamOk = false;
let lastDetections = [];
let totalEvents = 0;
let maxCount = 1;
let prevAmb = 0;
let processedEventIds = new Set(); // Track which events we've already processed
let isFirstPoll = true; // Track if this is the first status poll

// ── Traffic Light State ────────────────────────────────────────────────────
let trafficLightState = "red"; // 'red', 'green', 'yellow'
let greenTimeout = null;
let yellowTimeout = null;
let lastAmbulanceTime = null;

function setTrafficLight(state) {
  // Remove all active states
  document.getElementById("lightRed").className = "light off";
  document.getElementById("lightYellow").className = "light off";
  document.getElementById("lightGreen").className = "light off";

  // Set the new state
  if (state === "red") {
    document.getElementById("lightRed").className = "light red";
  } else if (state === "yellow") {
    document.getElementById("lightYellow").className = "light yellow";
  } else if (state === "green") {
    document.getElementById("lightGreen").className = "light green";
  }

  trafficLightState = state;
}

function handleAmbulanceDetection() {
  // Cancel any pending timeouts
  if (greenTimeout) clearTimeout(greenTimeout);
  if (yellowTimeout) clearTimeout(yellowTimeout);

  // Set to green immediately
  setTrafficLight("green");
  lastAmbulanceTime = Date.now();

  // Schedule transition to yellow after 5 seconds
  greenTimeout = setTimeout(() => {
    setTrafficLight("yellow");

    // Schedule transition to red after 2 seconds
    yellowTimeout = setTimeout(() => {
      setTrafficLight("red");
    }, 2000);
  }, 5000);
}

// ── Clock ──────────────────────────────────────────────────────────────────
function tick() {
  const now = new Date();
  document.getElementById("clock").textContent = now.toTimeString().slice(0, 8);
  document.getElementById("tsInfo").textContent = now
    .toTimeString()
    .slice(0, 8);
}
setInterval(tick, 1000);
tick();

// ── Stream load/error ──────────────────────────────────────────────────────
function handleStreamLoad() {
  if (!streamOk) {
    streamOk = true;
    document.getElementById("connectOverlay").style.display = "none";
  }
}

function handleStreamError() {
  streamOk = false;
  document.getElementById("connectOverlay").style.display = "flex";
  // Retry after 2s
  setTimeout(() => {
    const img = document.getElementById("streamImg");
    img.src = "/stream?" + Date.now();
  }, 2000);
}

// ── Status polling ─────────────────────────────────────────────────────────
async function pollStatus() {
  try {
    const r = await fetch("/status");
    if (!r.ok) return;
    const d = await r.json();

    // Status dot
    const dot = document.getElementById("statusDot");
    const stxt = document.getElementById("statusText");
    dot.className = "status-dot " + (d.stream_status || "connecting");
    const labels = {
      live: "EN VIVO",
      connecting: "CONECTANDO",
      error: "ERROR",
    };
    stxt.textContent = labels[d.stream_status] || "DESCONECTADO";

    // Stats
    document.getElementById("statTotal").textContent = d.total_detections ?? 0;
    document.getElementById("statFps").textContent = d.fps ?? 0;
    document.getElementById("statAmb").textContent =
      d.detection_count?.ambulance ?? 0;
    document.getElementById("statUp").textContent = d.uptime ?? 0;
    document.getElementById("fpsFrames").textContent = d.frame_count ?? 0;
    document.getElementById("modelLabel").textContent = d.model ?? "—";
    document.getElementById("modelBadge").textContent = d.model ?? "—";
    document.getElementById("frameInfo").textContent =
      "FRAME " + (d.frame_count ?? "--");

    // Vehicle bars
    const cnts = d.detection_count || {};
    const order = ["car", "ambulance", "bus", "truck", "motorcycle"];
    maxCount = Math.max(1, ...order.map((k) => cnts[k] || 0));

    const setBar = (id, barId, key) => {
      const v = cnts[key] || 0;
      document.getElementById(id).textContent = v;
      document.getElementById(barId).style.width = (v / maxCount) * 100 + "%";
    };
    setBar("cntCar", "barCar", "car");
    setBar("cntAmb", "barAmb", "ambulance");
    setBar("cntBus", "barBus", "bus");
    setBar("cntTruck", "barTruck", "truck");
    setBar("cntMoto", "barMoto", "motorcycle");

    // Ambulance alert flash
    const newAmb = cnts.ambulance || 0;
    if (newAmb > prevAmb && !isFirstPoll) {
      document.getElementById("videoPanel").classList.remove("alert-ambulance");
      void document.getElementById("videoPanel").offsetWidth;
      document.getElementById("videoPanel").classList.add("alert-ambulance");
      // Trigger ambulance detection for traffic light
      handleAmbulanceDetection();
    }
    prevAmb = newAmb;
    isFirstPoll = false;

    // Event feed — add new detections
    const dets = d.last_detections || [];
    if (
      dets.length &&
      JSON.stringify(dets) !== JSON.stringify(lastDetections)
    ) {
      lastDetections = dets;
      addEvents(dets);
    }
  } catch (e) {
    /* silent */
  }
}

function addEvents(dets) {
  const feed = document.getElementById("eventFeed");

  // Clear placeholder
  const placeholder = feed.querySelector('[style*="ESPERANDO"]');
  if (placeholder) placeholder.remove();

  const now = new Date().toTimeString().slice(0, 8);
  dets.forEach((det) => {
    // Create unique ID for this detection (label + bbox coordinates hash)
    const eventId = `${det.label}-${Math.round(det.bbox[0])}-${Math.round(det.bbox[1])}-${Math.round(det.bbox[2])}-${Math.round(det.bbox[3])}`;

    // Only add if we haven't already processed this exact detection
    if (!processedEventIds.has(eventId)) {
      processedEventIds.add(eventId);

      const item = document.createElement("div");
      item.className = "event-item";
      const clr = COLORS[det.label] || "var(--accent)";
      item.style.setProperty("--clr", clr);
      item.style.borderLeftColor = clr;
      item.innerHTML = `
        <span class="event-time">${now}</span>
        <span class="event-label">${ICONS[det.label] || ""} ${det.label}</span>
        <span class="event-conf">${(det.conf * 100).toFixed(0)}%</span>
      `;
      feed.insertBefore(item, feed.firstChild);
    }

    // Keep max 60 events
    while (feed.children.length > 60) feed.removeChild(feed.lastChild);
  });
}

// ── Initialization ────────────────────────────────────────────────────────
function initializeApp() {
  // Set traffic light to red on startup
  setTrafficLight("red");

  // Clear event feed on startup
  const feed = document.getElementById("eventFeed");
  feed.innerHTML =
    "<div style=\"text-align:center; color:var(--text-dim); font-size:11px; padding:20px; font-family:'Share Tech Mono',monospace; letter-spacing:2px;\">ESPERANDO DETECCIONES...</div>";

  // Reset vehicle counters to 0
  document.getElementById("cntCar").textContent = "0";
  document.getElementById("cntAmb").textContent = "0";
  document.getElementById("cntBus").textContent = "0";
  document.getElementById("cntTruck").textContent = "0";
  document.getElementById("cntMoto").textContent = "0";

  // Reset vehicle bars
  document.getElementById("barCar").style.width = "0%";
  document.getElementById("barAmb").style.width = "0%";
  document.getElementById("barBus").style.width = "0%";
  document.getElementById("barTruck").style.width = "0%";
  document.getElementById("barMoto").style.width = "0%";

  // Reset stats
  document.getElementById("statTotal").textContent = "0";
  document.getElementById("statAmb").textContent = "0";

  // Reset processed events tracking
  processedEventIds.clear();

  // Reset server counters
  fetch("/reset", { method: "POST" }).catch((e) =>
    console.log("Reset endpoint called"),
  );
}

// Initialize on load
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initializeApp);
} else {
  initializeApp();
}

// Poll every 1s
setInterval(pollStatus, 1000);
