const fileInput = document.getElementById("fileInput");
const dropZone = document.getElementById("dropZone");
const fileName = document.getElementById("fileName");
const runBtn = document.getElementById("runBtn");
const progressFill = document.getElementById("progressFill");
const progressText = document.getElementById("progressText");
const statusText = document.getElementById("statusText");
const logBox = document.getElementById("logBox");

const caseIdText = document.getElementById("caseId");
const inputFileNameText = document.getElementById("inputFileName");
const inputTypeText = document.getElementById("inputType");
const modelNameText = document.getElementById("modelName");
const checkpointNameText = document.getElementById("checkpointName");

const predictionMaskLink = document.getElementById("predictionMaskLink");
const femurStlLink = document.getElementById("femurStlLink");
const tibiaStlLink = document.getElementById("tibiaStlLink");
const downloadResultsBtn = document.getElementById("downloadResultsBtn");

let selectedFile = null;
let latestResultUrls = null;
let activePollInterval = null;
let lastLoggedStatusKey = "";


function log(message) {
  const currentText = logBox.textContent || "";
  const lines = currentText.split("\n");
  const lastLine = lines[lines.length - 1];

  // Prevent duplicate consecutive log lines.
  if (lastLine === message) {
    return;
  }

  logBox.textContent += `\n${message}`;
  logBox.scrollTop = logBox.scrollHeight;
}


function logStatus(status, message) {
  const safeStatus = status || "Status";
  const safeMessage = message || "";
  const key = `${safeStatus}:${safeMessage}`;

  // Prevent repeated polling messages from flooding the log.
  if (key === lastLoggedStatusKey) {
    return;
  }

  lastLoggedStatusKey = key;
  log(`${safeStatus}: ${safeMessage}`);
}


function clearActivePolling() {
  if (activePollInterval) {
    clearInterval(activePollInterval);
    activePollInterval = null;
  }
}


function getInputType(filename) {
  const lower = filename.toLowerCase();

  if (lower.endsWith(".nii.gz")) return "NIfTI compressed (.nii.gz)";
  if (lower.endsWith(".nii")) return "NIfTI (.nii)";
  if (lower.endsWith(".zip")) return "DICOM ZIP";
  return "Unknown";
}


function fullBackendUrl(path) {
  if (!path || path === "#") return "#";
  return `http://127.0.0.1:8000${path}`;
}


function resetOutputLinks() {
  latestResultUrls = null;

  predictionMaskLink.href = "#";
  predictionMaskLink.textContent = "Not available";

  femurStlLink.href = "#";
  femurStlLink.textContent = "Not available";

  tibiaStlLink.href = "#";
  tibiaStlLink.textContent = "Not available";

  downloadResultsBtn.disabled = true;
}


function updateOutputLinks(status) {
  latestResultUrls = {
    predictionMask: fullBackendUrl(status.prediction_mask_url),
    femurStl: fullBackendUrl(status.femur_stl_url),
    tibiaStl: fullBackendUrl(status.proximal_tibia_stl_url),
    zip: `http://127.0.0.1:8000/download/${status.case_id}`,
  };

  predictionMaskLink.href = latestResultUrls.predictionMask;
  predictionMaskLink.textContent = "prediction_mask.nii.gz";

  femurStlLink.href = latestResultUrls.femurStl;
  femurStlLink.textContent = "femur.stl";

  tibiaStlLink.href = latestResultUrls.tibiaStl;
  tibiaStlLink.textContent = "proximal_tibia.stl";

  downloadResultsBtn.disabled = false;
}


function resetForNewFile(file) {
  selectedFile = file;
  latestResultUrls = null;
  lastLoggedStatusKey = "";

  fileName.textContent = file.name;
  inputFileNameText.textContent = file.name;
  inputTypeText.textContent = getInputType(file.name);

  caseIdText.textContent = "Not started";
  statusText.textContent = "File Loaded";

  progressFill.style.width = "0%";
  progressText.textContent = "0%";

  runBtn.disabled = false;
  runBtn.textContent = "Run Segmentation";

  resetOutputLinks();
}


function setFile(file) {
  clearActivePolling();
  resetForNewFile(file);
  log(`Loaded input: ${file.name}`);
}


dropZone.addEventListener("click", () => fileInput.click());


fileInput.addEventListener("change", (event) => {
  const file = event.target.files[0];
  if (file) setFile(file);
});


dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("dragover");
});


dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("dragover");
});


dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("dragover");

  const file = event.dataTransfer.files[0];
  if (file) setFile(file);
});


runBtn.addEventListener("click", async () => {
  if (!selectedFile) {
    log("No input selected.");
    statusText.textContent = "Waiting for File";
    return;
  }

  clearActivePolling();
  lastLoggedStatusKey = "";

  runBtn.disabled = true;
  runBtn.textContent = "Processing...";

  statusText.textContent = "Uploading";
  progressFill.style.width = "5%";
  progressText.textContent = "5%";

  caseIdText.textContent = "Uploading...";
  inputFileNameText.textContent = selectedFile.name;
  inputTypeText.textContent = getInputType(selectedFile.name);
  modelNameText.textContent = "Dataset002 Human Femur + Proximal Tibia";
  checkpointNameText.textContent = "checkpoint_best.pth";

  resetOutputLinks();
  log("Uploading file to backend...");

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const startResponse = await fetch("http://127.0.0.1:8000/segment/start", {
      method: "POST",
      body: formData,
    });

    if (!startResponse.ok) {
      const errorText = await startResponse.text();
      throw new Error(errorText);
    }

    const startResult = await startResponse.json();
    const caseId = startResult.case_id;

    caseIdText.textContent = caseId;

    log(`Case ID: ${caseId}`);
    log(startResult.message || "File uploaded. Segmentation started.");

    statusText.textContent = "Queued";
    progressFill.style.width = `${startResult.progress || 10}%`;
    progressText.textContent = `${startResult.progress || 10}%`;

    const prettyStatus = {
      queued: "Queued",
      converting_input: "Converting DICOM",
      cropping_input: "Cropping CT",
      predicting: "Running nnU-Net",
      converting: "Generating STL",
      complete: "Complete",
      error: "Error",
    };

    activePollInterval = setInterval(async () => {
      try {
        const statusResponse = await fetch(
          `http://127.0.0.1:8000/segment/status/${caseId}`
        );

        if (!statusResponse.ok) {
          throw new Error("Failed to get job status.");
        }

        const status = await statusResponse.json();

        if (status.case_id) {
          caseIdText.textContent = status.case_id;
        }

        if (status.filename) {
          inputFileNameText.textContent = status.filename;
        }

        const displayStatus = prettyStatus[status.status] || status.status || "Processing";

        statusText.textContent = displayStatus;

        const progress = status.progress || 0;
        progressFill.style.width = `${progress}%`;
        progressText.textContent = `${progress}%`;

        logStatus(displayStatus, status.message);

        if (status.status === "complete") {
          clearActivePolling();

          log("Segmentation complete.");
          log(`Prediction mask: ${status.prediction_mask_url}`);
          log(`Femur STL: ${status.femur_stl_url}`);
          log(`Proximal tibia STL: ${status.proximal_tibia_stl_url}`);

          updateOutputLinks(status);

          if (window.loadSegmentationSTL) {
            await window.loadSegmentationSTL(
              status.femur_stl_url,
              status.proximal_tibia_stl_url
            );
          }

          statusText.textContent = "Segmentation Complete";
          progressFill.style.width = "100%";
          progressText.textContent = "100%";

          runBtn.disabled = false;
          runBtn.textContent = "Run Segmentation";
        }

        if (status.status === "error") {
          clearActivePolling();

          statusText.textContent = "Error";
          log(`Error: ${status.error || status.message}`);

          runBtn.disabled = false;
          runBtn.textContent = "Run Segmentation";
        }
      } catch (pollError) {
        clearActivePolling();

        statusText.textContent = "Error";
        log(`Polling error: ${pollError.message}`);

        runBtn.disabled = false;
        runBtn.textContent = "Run Segmentation";
      }
    }, 5000);
  } catch (error) {
    console.error(error);

    statusText.textContent = "Error";
    log(`Error: ${error.message}`);

    runBtn.disabled = false;
    runBtn.textContent = "Run Segmentation";
  }
});


downloadResultsBtn.addEventListener("click", () => {
  if (!latestResultUrls || !latestResultUrls.zip) {
    log("No results available for download.");
    return;
  }

  log("Downloading result ZIP...");

  const link = document.createElement("a");
  link.href = latestResultUrls.zip;
  link.download = "segmentation_results.zip";
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
});