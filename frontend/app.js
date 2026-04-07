/**
 * Frontend logic for Classifier AX.
 * Manages API calls, state management, autocomplete DOM injection, and Chart.js SHAP visualizations.
 */

const API_BASE = "http://127.0.0.1:8000";

// State
let validSymptoms = [];
let selectedSymptoms = {}; // "symptom_name": 1 (present)
let shapChartInstance = null;

// DOM Elements
const searchInput = document.getElementById("symptom-search");
const autocompleteList = document.getElementById("autocomplete-list");
const selectedList = document.getElementById("selected-list");
const emptyState = document.getElementById("empty-state");
const predictBtn = document.getElementById("predict-btn");

// Results Elements
const resultsPlaceholder = document.getElementById("placeholder");
const resultsContent = document.getElementById("results-content");
const diseaseNameEl = document.getElementById("disease-name");
const diseaseCategoryEl = document.getElementById("disease-category");
const confPercentageEl = document.getElementById("conf-percentage");
const confFillEl = document.getElementById("conf-fill");
const latencyValEl = document.getElementById("latency-val");
const decisionRuleEl = document.getElementById("decision-rule");
const ctx = document.getElementById("shapChart").getContext("2d");

// ── Initialization ──────────────────────────────────────────────────────────
async function init() {
    try {
        const res = await fetch(`${API_BASE}/symptoms`);
        if (!res.ok) throw new Error("API not accessible");
        const data = await res.json();
        validSymptoms = data.symptoms;
        console.log(`Loaded ${validSymptoms.length} symptoms from API.`);
        searchInput.disabled = false;
        searchInput.placeholder = "Search symptoms... (e.g. fever)";
    } catch (e) {
        console.error("Failed to fetch symptoms:", e);
        searchInput.placeholder = "Error connecting to backend...";
        searchInput.disabled = true;
    }
}

// ── Event Listeners ────────────────────────────────────────────────────────


searchInput.addEventListener("input", function() {
    const val = this.value.trim().toLowerCase();
    closeAllLists();
    if (!val) return false;
    
    // Filter matching symptoms (limit to top 10 for performance)
    const matches = validSymptoms.filter(s => s.toLowerCase().includes(val)).slice(0, 10);
    
    if (matches.length > 0) {
        autocompleteList.style.display = "block";
        matches.forEach(match => {
            const div = document.createElement("div");
            // Bold the matching text
            const regex = new RegExp(`(${val})`, "gi");
            div.innerHTML = match.replace(regex, "<strong>$1</strong>");
            
            div.addEventListener("click", () => {
                addSymptom(match);
                searchInput.value = "";
                closeAllLists();
            });
            autocompleteList.appendChild(div);
        });
    }
});

// Hide autocomplete on outside click
document.addEventListener("click", function (e) {
    if (e.target !== searchInput) {
        closeAllLists();
    }
});

predictBtn.addEventListener("click", processPrediction);

// ── Functions ──────────────────────────────────────────────────────────────
function closeAllLists() {
    autocompleteList.innerHTML = "";
    autocompleteList.style.display = "none";
}

function addSymptom(name) {
    if (selectedSymptoms.hasOwnProperty(name)) {
        // Just update state if it exists
        selectedSymptoms[name] = 1;
        renderTags();
        return;
    }
    
    selectedSymptoms[name] = 1;
    renderTags();
    validateForm();
}

function removeSymptom(name) {
    delete selectedSymptoms[name];
    renderTags();
    validateForm();
}

function renderTags() {
    selectedList.innerHTML = "";
    
    const keys = Object.keys(selectedSymptoms);
    
    if (keys.length === 0) {
        emptyState.style.display = "block";
    } else {
        emptyState.style.display = "none";
        
        keys.forEach(k => {
            const val = selectedSymptoms[k];
            const li = document.createElement("li");
            li.className = `tag tag-present`;
            
            const stateIcon = "+";
            
            // Format nice text replacing underscores
            const text = k.replace(/_/g, " ");
            
            li.innerHTML = `
                ${text} <strong>(${stateIcon})</strong>
                <button type="button" class="tag-remove">&times;</button>
            `;
            
            li.querySelector(".tag-remove").addEventListener("click", (e) => {
                e.stopPropagation();
                removeSymptom(k);
            });
            
            selectedList.appendChild(li);
        });
    }
}

function validateForm() {
    // Only allow prediction if at least one PRESENT symptom exists
    const hasPresent = Object.values(selectedSymptoms).includes(1);
    predictBtn.disabled = !hasPresent;
}

async function processPrediction() {
    // UI state
    predictBtn.classList.add("loading");
    predictBtn.disabled = true;
    
    try {
        const response = await fetch(`${API_BASE}/predict`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ symptoms: selectedSymptoms })
        });
        
        if (!response.ok) throw new Error("Diagnostics API failed.");
        
        const data = await response.json();
        renderResults(data);
        
    } catch (e) {
        console.error(e);
        alert("Inference failed! Verify backend is running on port 8000.");
    } finally {
        predictBtn.classList.remove("loading");
        predictBtn.disabled = false;
    }
}

function renderResults(data) {
    // UI Panels
    resultsPlaceholder.classList.add("hide");
    resultsContent.classList.remove("hide");
    
    // Core data
    diseaseNameEl.textContent = data.disease;
    diseaseCategoryEl.textContent = data.category.replace(/_/g, " ");
    
    // Confidence (calibrated + sparsity-adjusted prob)
    const probPct = Math.round(data.probability * 100);
    confPercentageEl.textContent = `${probPct}%`;
    // Delay width to allow css animation to trigger
    setTimeout(() => {
        confFillEl.style.width = `${probPct}%`;
    }, 50);
    
    // Color confidence bar by level
    if (probPct < 40) {
        confFillEl.className = "progress-bar-fill conf-low";
    } else if (probPct < 70) {
        confFillEl.className = "progress-bar-fill conf-medium";
    } else {
        confFillEl.className = "progress-bar-fill conf-high";
    }
    
    // Low-confidence warning banner
    let warningEl = document.getElementById("low-conf-warning");
    if (!warningEl) {
        warningEl = document.createElement("div");
        warningEl.id = "low-conf-warning";
        warningEl.className = "low-confidence-warning";
        // Insert after the status badge
        const badge = resultsContent.querySelector(".status-badge");
        badge.parentNode.insertBefore(warningEl, badge.nextSibling);
    }
    
    if (probPct < 50) {
        warningEl.classList.remove("hide");
        warningEl.innerHTML = `
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
                <line x1="12" y1="9" x2="12" y2="13"></line>
                <line x1="12" y1="17" x2="12.01" y2="17"></line>
            </svg>
            <span>Low confidence — add more symptoms for a reliable diagnosis</span>
        `;
    } else {
        warningEl.classList.add("hide");
    }
    
    latencyValEl.textContent = data.latency_ms;
    decisionRuleEl.textContent = data.decision_path;

    // Disease Description
    document.getElementById("disease-desc-text").textContent = data.disease_description || "Description not available.";

    // Symptom Severities
    const sevContainer = document.getElementById("severity-tags-container");
    sevContainer.innerHTML = "";
    let totalSev = 0;
    
    if (data.symptom_severities) {
        for (const [symp, weight] of Object.entries(data.symptom_severities)) {
            totalSev += weight;
            const tag = document.createElement("div");
            tag.className = "sev-tag";
            const readableSymp = symp.replace(/_/g, " ");
            tag.innerHTML = `<span>${readableSymp}</span> <span class="badge">Weight: ${weight}</span>`;
            sevContainer.appendChild(tag);
        }
    }
    document.getElementById("total-severity-val").textContent = totalSev;

    // SHAP Chart Rendering
    renderShapChart(data.shap_values);
}

function renderShapChart(shapData) {
    const features = Object.keys(shapData);
    const values = Object.values(shapData);
    
    // Determine colors (red for pushes positive, blue for pushes negative)
    const colors = values.map(v => v > 0 ? "rgba(244, 63, 94, 0.8)" : "rgba(56, 189, 248, 0.8)");
    const borderColors = values.map(v => v > 0 ? "rgba(244, 63, 94, 1)" : "rgba(56, 189, 248, 1)");

    if (shapChartInstance) {
        shapChartInstance.destroy();
    }

    // Modern dark theme defaults
    Chart.defaults.color = 'rgba(255, 255, 255, 0.6)';
    Chart.defaults.font.family = 'Outfit';

    shapChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: features.map(f => f.replace(/_/g, " ")),
            datasets: [{
                label: 'SHAP Impact Magnitude',
                data: values,
                backgroundColor: colors,
                borderColor: borderColors,
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y', // Horizontal bar chart
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.9)',
                    titleFont: { size: 14 },
                    padding: 12,
                    cornerRadius: 8,
                    displayColors: false
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    title: { display: true, text: 'Impact on output probability' }
                },
                y: {
                    grid: { display: false }
                }
            }
        }
    });
}

// ── Boot ────────────────────────────────────────────────────────────────────
init();
