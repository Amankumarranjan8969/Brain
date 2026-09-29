// ============================================================
//  NeuroScan AI — Frontend logic
// ============================================================
const API_BASE = ""; // same-origin

const fileInput     = document.getElementById("fileInput");
const dropzone      = document.getElementById("dropzone");
const dropzoneInner = document.getElementById("dropzoneInner");
const previewWrap   = document.getElementById("previewWrap");
const previewImg    = document.getElementById("previewImg");
const clearBtn      = document.getElementById("clearBtn");
const analyzeBtn    = document.getElementById("analyzeBtn");

const idleState     = document.getElementById("idleState");
const loadingState  = document.getElementById("loadingState");
const resultState   = document.getElementById("resultState");

const diagnosisName = document.getElementById("diagnosisName");
const confidenceBar = document.getElementById("confidenceBar");
const confidenceVal = document.getElementById("confidenceVal");
const resultOriginal= document.getElementById("resultOriginal");
const resultGradcam = document.getElementById("resultGradcam");
const probsList     = document.getElementById("probsList");

let currentFile = null;

fileInput.addEventListener("change", (e) => {
  if (e.target.files && e.target.files[0]) setFile(e.target.files[0]);
});

["dragenter", "dragover"].forEach(evt => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });
});
["dragleave", "drop"].forEach(evt => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  });
});
dropzone.addEventListener("drop", (e) => {
  const f = e.dataTransfer.files?.[0];
  if (f) setFile(f);
});

function setFile(file) {
  if (!file.type.startsWith("image/")) {
    alert("Please upload an image file (JPG or PNG).");
    return;
  }
  currentFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    previewWrap.classList.remove("hidden");
    dropzoneInner.classList.add("hidden");
  };
  reader.readAsDataURL(file);
  analyzeBtn.disabled = false;
  showState("idle");
}

clearBtn.addEventListener("click", (e) => {
  e.preventDefault();
  currentFile = null;
  fileInput.value = "";
  previewImg.src = "";
  previewWrap.classList.add("hidden");
  dropzoneInner.classList.remove("hidden");
  analyzeBtn.disabled = true;
  showState("idle");
});

analyzeBtn.addEventListener("click", async () => {
  if (!currentFile) return;
  showState("loading");

  const formData = new FormData();
  formData.append("file", currentFile);

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Server error: ${res.status}`);
    }
    const data = await res.json();
    renderResult(data);
  } catch (err) {
    console.error(err);
    alert("Analysis failed: " + err.message);
    showState("idle");
  }
});

function renderResult(data) {
  diagnosisName.textContent = prettify(data.predicted_class);
  confidenceVal.textContent = data.confidence.toFixed(2) + "%";
  confidenceBar.style.width = "0%";
  setTimeout(() => { confidenceBar.style.width = data.confidence + "%"; }, 60);

  const cls = data.predicted_class.toLowerCase();
  const colorMap = {
    glioma:     { main: "#dc2626", glow: "rgba(220,38,38,0.45)" },
    meningioma: { main: "#ca8a04", glow: "rgba(202,138,4,0.45)" },
    notumor:    { main: "#16a34a", glow: "rgba(22,163,74,0.45)" },
    pituitary:  { main: "#0369a1", glow: "rgba(3,105,161,0.45)" },
  };
  const c = colorMap[cls] || colorMap.pituitary;
  diagnosisName.style.color = c.main;
  confidenceBar.style.background = `linear-gradient(90deg, ${c.main}, ${lighten(c.main)})`;
  confidenceBar.style.boxShadow = `0 0 14px ${c.glow}`;

  resultOriginal.src = data.original_image;
  resultGradcam.src = data.gradcam_image || data.original_image;

  const entries = Object.entries(data.probabilities).sort((a, b) => b[1] - a[1]);
  probsList.innerHTML = entries.map(([label, val]) => `
    <div class="prob-row">
      <span class="prob-label">${prettify(label)}</span>
      <div class="prob-track">
        <div class="prob-fill" style="width:0%" data-w="${val}"></div>
      </div>
      <span class="prob-value">${val.toFixed(2)}%</span>
    </div>
  `).join("");

  requestAnimationFrame(() => {
    document.querySelectorAll(".prob-fill").forEach(el => {
      el.style.width = el.dataset.w + "%";
    });
  });

  showState("result");
}

function showState(which) {
  idleState.classList.add("hidden");
  loadingState.classList.add("hidden");
  resultState.classList.add("hidden");
  if (which === "idle")    idleState.classList.remove("hidden");
  if (which === "loading") loadingState.classList.remove("hidden");
  if (which === "result")  resultState.classList.remove("hidden");
}

function prettify(s) { return s.charAt(0).toUpperCase() + s.slice(1); }

function lighten(hex) {
  const n = parseInt(hex.slice(1), 16);
  const r = Math.min(255, ((n >> 16) & 0xff) + 40);
  const g = Math.min(255, ((n >> 8) & 0xff) + 40);
  const b = Math.min(255, (n & 0xff) + 40);
  return `rgb(${r},${g},${b})`;
}