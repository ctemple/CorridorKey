/**
 * CorridorKey Web UI — Frontend Logic
 * Handles upload, parameter management, SSE progress, and download.
 */

(function () {
  "use strict";

  // ── State ────────────────────────────────────
  let currentJobId = null;
  let currentFileName = null;
  let inputBitrate = 0;  // probed from uploaded video (bps)
  let eventSource = null;

  // ── DOM refs ─────────────────────────────────
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => document.querySelectorAll(sel);

  const uploadZone = $("#upload-zone");
  const uploadText = $("#upload-text");
  const fileName = $("#file-name");
  const fileInput = $("#file-input");
  const processBtn = $("#process-btn");
  const form = $("#params-form");
  const jobList = $("#job-list");
  const jobsSection = $("#jobs-section");
  const progressBar = $("#progress-bar");
  const progressFill = $("#progress-fill");
  const progressStats = $("#progress-stats");
  const progressLabel = $("#progress-label");
  const downloadLink = $("#download-link");
  const toastContainer = $("#toast-container");
  const bitratePreset = $("#output_bitrate_preset");
  const bitrateCustom = $("#output_bitrate_custom");
  const inputBitrateLabel = $("#input-bitrate-label");

  // ── Bitrate preset dropdown ──────────────────
  if (bitratePreset && bitrateCustom) {
    bitratePreset.addEventListener("change", () => {
      if (bitratePreset.value === "-1") {
        bitrateCustom.style.display = "block";
        bitrateCustom.focus();
      } else {
        bitrateCustom.style.display = "none";
        bitrateCustom.value = "";
      }
    });
  }

  // ── Parameter value displays ─────────────────
  const despillRange = $("#despill_strength");
  const despillValue = $("#despill-strength-value");
  if (despillRange) {
    despillRange.addEventListener("input", () => {
      despillValue.textContent = despillRange.value;
    });
  }

  // ── File Upload ──────────────────────────────

  uploadZone.addEventListener("click", () => fileInput.click());

  uploadZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadZone.classList.add("drag-over");
  });

  uploadZone.addEventListener("dragleave", () => {
    uploadZone.classList.remove("drag-over");
  });

  uploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadZone.classList.remove("drag-over");
    const files = e.dataTransfer.files;
    if (files.length > 0) handleFile(files[0]);
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) handleFile(fileInput.files[0]);
  });

  async function handleFile(file) {
    const validExts = [".mp4", ".mov", ".avi", ".mkv", ".webm"];
    const ext = "." + file.name.split(".").pop().toLowerCase();
    if (!validExts.includes(ext)) {
      showToast(`Unsupported format. Allowed: ${validExts.join(", ")}`, "error");
      return;
    }

    uploadText.textContent = "Uploading…";
    uploadZone.classList.add("has-file");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const resp = await fetch("/api/upload", { method: "POST", body: formData });
      if (!resp.ok) {
        const err = await resp.json();
        throw new Error(err.detail || "Upload failed");
      }
      const job = await resp.json();
      currentJobId = job.job_id;
      currentFileName = file.name;
      inputBitrate = job.input_bitrate || 0;
      fileName.textContent = `✓ ${file.name}`;
      processBtn.disabled = false;
      // Show probed input bitrate next to the dropdown
      if (inputBitrate > 0) {
        inputBitrateLabel.textContent = `Input: ${formatBitrate(inputBitrate)}`;
      } else {
        inputBitrateLabel.textContent = "";
      }
      showToast("Upload complete — configure parameters and click Process", "success");
      refreshJobs();
    } catch (err) {
      showToast(err.message, "error");
      uploadText.textContent = "Drag & drop or click to upload";
      uploadZone.classList.remove("has-file");
    }
  }

  // ── Process ──────────────────────────────────

  processBtn.addEventListener("click", async () => {
    if (!currentJobId) return;

    const formData = new FormData();
    formData.append("screen_color", $("#screen_color").value);
    formData.append("input_is_linear", $("#input_is_linear").checked);
    formData.append("despill_strength", $("#despill_strength").value);
    formData.append("auto_despeckle", $("#auto_despeckle").checked);
    formData.append("despeckle_size", $("#despeckle_size").value);
    formData.append("image_size", $("#image_size").value);
    formData.append("refiner_scale", $("#refiner_scale").value);
    formData.append("generate_comp", $("#generate_comp").checked);
    formData.append("gpu_post_processing", $("#gpu_post_processing").checked);
    formData.append("output_bitrate", getOutputBitrate());
    formData.append("mask_mode", getMaskMode());

    processBtn.disabled = true;
    processBtn.textContent = "Starting…";

    try {
      const resp = await fetch(`/api/jobs/${currentJobId}/start`, {
        method: "POST",
        body: formData,
      });
      if (!resp.ok) {
        const err = await resp.json();
        throw new Error(err.detail || "Failed to start job");
      }
      showToast("Job queued — processing will begin shortly", "success");
      connectSSE(currentJobId);
      refreshJobs();
    } catch (err) {
      showToast(err.message, "error");
      processBtn.disabled = false;
      processBtn.textContent = "Process Video";
    }
  });

  // ── SSE (progress updates) ───────────────────

  function connectSSE(jobId) {
    if (eventSource) {
      eventSource.close();
      eventSource = null;
    }

    eventSource = new EventSource(`/api/events/${jobId}`);

    eventSource.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        updateProgress(data);

        if (data.state === "completed" || data.state === "failed") {
          eventSource.close();
          eventSource = null;
          processBtn.disabled = false;
          processBtn.textContent = "Process Video";

          if (data.state === "completed") {
            showToast("Processing complete! Download ready.", "success");
          } else {
            showToast(`Processing failed: ${data.error || "Unknown error"}`, "error");
            processBtn.disabled = false;
          }
          refreshJobs();
        }
      } catch (_) {}
    };

    eventSource.onerror = () => {
      // SSE will auto-reconnect; if it stays broken, clean up
      setTimeout(() => {
        if (eventSource && eventSource.readyState === EventSource.CLOSED) {
          eventSource = null;
        }
      }, 5000);
    };
  }

  function updateProgress(data) {
    const jobEl = $(`.job-item[data-job-id="${data.job_id}"]`);
    if (jobEl) {
      updateJobItem(jobEl, data);
    }

    // Show progress bar for the active job
    if (data.state === "processing") {
      progressBar.style.display = "block";
      const pct = data.total_frames > 0
        ? Math.round((data.current_frame / data.total_frames) * 100)
        : 0;
      progressFill.style.width = `${pct}%`;
      progressLabel.textContent = data.sub_step_label || "Processing…";
      progressStats.textContent = data.total_frames > 0
        ? `${data.current_frame} / ${data.total_frames} frames`
        : "";
    } else if (data.state === "completed" || data.state === "failed") {
      progressBar.style.display = "none";
    }
  }

  // ── Job List ─────────────────────────────────

  async function refreshJobs() {
    try {
      const resp = await fetch("/api/jobs");
      if (!resp.ok) return;
      const jobs = await resp.json();
      renderJobs(jobs);
    } catch (_) {}
  }

  function renderJobs(jobs) {
    if (jobs.length === 0) {
      jobsSection.style.display = "none";
      return;
    }
    jobsSection.style.display = "block";
    jobList.innerHTML = jobs
      .reverse()
      .map(
        (j) => `
      <li class="job-item" data-job-id="${j.job_id}">
        <span class="job-status-dot ${j.state}"></span>
        <div class="job-info">
          <div class="job-name">${escHtml(j.original_filename || j.job_id)}</div>
          <div class="job-meta">${statusLabel(j)}</div>
          ${j.state === "processing" ? progressBarHtml(j) : ""}
        </div>
        <div class="job-actions">
          ${j.state === "completed"
            ? `<a class="btn-sm download" href="/api/jobs/${j.job_id}/download" download>⬇ Download</a>`
            : ""}
        </div>
      </li>`
      )
      .join("");
  }

  function updateJobItem(el, data) {
    const dot = el.querySelector(".job-status-dot");
    const meta = el.querySelector(".job-meta");
    const actions = el.querySelector(".job-actions");

    if (dot) {
      dot.className = `job-status-dot ${data.state}`;
    }
    if (meta) {
      meta.textContent = statusLabel(data);
    }
    if (actions && data.state === "completed") {
      actions.innerHTML = `<a class="btn-sm download" href="/api/jobs/${data.job_id}/download" download>⬇ Download</a>`;
    }
  }

  function statusLabel(j) {
    switch (j.state) {
      case "uploaded":  return "Uploaded — awaiting configuration";
      case "queued":    return "Queued — waiting for GPU…";
      case "processing":
        const pct = j.total_frames > 0 ? Math.round((j.current_frame / j.total_frames) * 100) : 0;
        return `${j.sub_step_label || "Processing"} (${pct}%)`;
      case "completed": return "✅ Complete";
      case "failed":    return `❌ Failed: ${j.error || "Unknown error"}`;
      default:          return j.state;
    }
  }

  function progressBarHtml(j) {
    const pct = j.total_frames > 0 ? Math.round((j.current_frame / j.total_frames) * 100) : 0;
    return `
      <div class="progress-bar-container">
        <div class="progress-bar-track">
          <div class="progress-bar-fill" style="width:${pct}%"></div>
        </div>
        <div class="progress-bar-stats">
          <span>${j.current_frame} / ${j.total_frames} frames</span>
          <span>${pct}%</span>
        </div>
      </div>`;
  }

  // ── Toast ────────────────────────────────────

  function showToast(message, type) {
    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    toast.textContent = message;
    toastContainer.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = "0";
      toast.style.transition = "opacity 0.3s";
      setTimeout(() => toast.remove(), 300);
    }, 4000);
  }

  function escHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function getOutputBitrate() {
    const preset = bitratePreset.value;
    if (preset === "-1") {
      // Custom: user enters kbps, convert to bps
      const kbps = parseInt(bitrateCustom.value, 10);
      if (isNaN(kbps) || kbps <= 0) return 0;
      return kbps * 1000;
    }
    return parseInt(preset, 10);
  }

  function formatBitrate(bps) {
    if (bps >= 1000000) {
      return (bps / 1000000).toFixed(1) + " Mbps";
    }
    if (bps >= 1000) {
      return (bps / 1000).toFixed(0) + " kbps";
    }
    return bps + " bps";
  }

  function getMaskMode() {
    const checked = document.querySelector('input[name="mask_mode"]:checked');
    return checked ? checked.value : "hybrid";
  }

  // ── Init ─────────────────────────────────────
  refreshJobs();
  // Poll for job list updates every 10 seconds
  setInterval(refreshJobs, 10000);
})();
