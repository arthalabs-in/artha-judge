const apiBase = window.location.origin;
let currentRecord = null;
let currentPdfUrl = null;
let currentPdfPage = 1;
let currentPdfPageCount = 1;
let currentPdfRecordId = null;
let activeQueueStatus = "";
let activeProgressJob = null;

function getApiKey() {
    return localStorage.getItem("apiKey") || localStorage.getItem("canvas_api_key") || "";
}

function getUserId() {
    const explicitUserId = localStorage.getItem("canvas_user_id");
    if (explicitUserId) return explicitUserId;
    const userName = localStorage.getItem("userName") || "demo_reviewer";
    return `user_${userName.toLowerCase().replace(/[^a-z0-9]/g, "_")}`;
}

function getReviewerRole() {
    return document.getElementById("roleSelect").value || "data_reviewer";
}

function setIdentityBanner() {
    document.getElementById("userDisplay").textContent =
        localStorage.getItem("canvas_user_id") || localStorage.getItem("userName") || "demo_reviewer";
    document.getElementById("apiStatus").textContent = getApiKey() ? "Configured" : "Demo/no key";
}

async function apiRequest(path, options = {}) {
    const apiKey = getApiKey();
    const headers = { ...options.headers };
    if (apiKey) {
        headers.Authorization = `Bearer ${apiKey}`;
    }
    if (!(options.body instanceof FormData) && !headers["Content-Type"]) {
        headers["Content-Type"] = "application/json";
    }
    const response = await fetch(`${apiBase}${path}`, { ...options, headers });
    if (!response.ok) {
        let detail = response.statusText;
        try {
            const payload = await response.json();
            detail = payload.detail || payload.error || detail;
        } catch (_) {}
        throw new Error(detail);
    }
    const contentType = response.headers.get("content-type") || "";
    if (contentType.includes("application/json")) {
        return response.json();
    }
    return response;
}

function showToast(message) {
    const toast = document.getElementById("toast");
    toast.textContent = message;
    toast.classList.add("show");
    window.clearTimeout(showToast.timer);
    showToast.timer = window.setTimeout(() => toast.classList.remove("show"), 3200);
}

function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>"']/g, (char) => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
    }[char]));
}

function humanize(value) {
    if (value === null || value === undefined || value === "") return "";
    return String(value)
        .replaceAll("_", " ")
        .replace(/\b\w/g, (char) => char.toUpperCase());
}

function humanizeAuditLabel(value) {
    const labels = {
        action_update: "Action Updated",
        duplicate_resolution: "Duplicate Resolved",
        field_update: "Field Updated",
        record_created: "Record Created",
        review_decision: "Review Decision",
        use_existing: "Use Existing",
        keep_both: "Keep Both",
        ai_value_wrong: "AI value wrong",
        missing_or_empty: "Missing or empty",
        source_verified: "Source verified",
        ambiguous_source: "Ambiguous source",
        format_cleanup: "Format cleanup",
        manual_correction: "Manual correction",
        pending_review: "Pending Review",
        needs_clarification: "Needs Clarification",
    };
    const key = String(value || "").trim();
    if (labels[key]) return labels[key];
    if (/^[a-z0-9]+(?:_[a-z0-9]+)+$/.test(key)) return humanize(key);
    return key;
}

function formatAuditValue(value) {
    if (value === null || value === undefined || value === "") return "";
    if (Array.isArray(value)) {
        return value.map(formatAuditValue).filter(Boolean).join(", ");
    }
    if (typeof value === "object") {
        const phrase = value.phrase || value.text || value.label || value.value || value.status || value.type || value.category;
        if (phrase) return formatAuditValue(phrase);
        return Object.entries(value)
            .filter(([, item]) => item !== null && item !== undefined && item !== "")
            .map(([key, item]) => `${humanizeAuditLabel(key)}: ${formatAuditValue(item)}`)
            .join("; ");
    }
    return humanizeAuditLabel(value);
}

function auditSummaryValue(payload) {
    if (!payload || typeof payload !== "object") return payload;
    if (payload.value !== undefined) return payload.value;
    if (payload.status !== undefined) return payload.status;
    if (payload.title !== undefined) {
        const owner = payload.responsible_department || payload.owner || payload.department;
        return owner ? `${payload.title} (${owner})` : payload.title;
    }
    if (payload.responsible_department !== undefined) return payload.responsible_department;
    if (payload.reason !== undefined) return payload.reason;
    return "";
}

function formatValue(value) {
    if (value === null || value === undefined || value === "") return "";
    if (Array.isArray(value)) {
        if (!value.length) return "";
        return value.map(formatValue).filter(Boolean).join(", ");
    }
    if (typeof value === "object") {
        const phrase = value.phrase || value.text || value.label || value.value || value.type || value.category;
        if (phrase) return formatValue(phrase);
        return Object.entries(value)
            .filter(([, item]) => item !== null && item !== undefined && item !== "")
            .map(([key, item]) => `${humanize(key)}: ${formatValue(item)}`)
            .join("; ");
    }
    return String(value);
}

function formatFieldDisplayValue(fieldKey, value) {
    if (fieldKey === "case_type" || fieldKey === "disposition") {
        return humanize(value);
    }
    return formatValue(value);
}

function statusPill(value) {
    return `<span class="status-pill">${escapeHtml(humanize(value || "pending_review"))}</span>`;
}

function setTaskProgress(payload = null) {
    const container = document.getElementById("taskStatus");
    const title = document.getElementById("taskStatusTitle");
    const message = document.getElementById("taskStatusMessage");
    const bar = document.getElementById("taskProgressBar");
    const stageList = document.getElementById("taskStageList");
    if (!container || !title || !message || !bar || !stageList) return;

    if (!payload) {
        container.className = "task-status idle";
        title.textContent = "No active task.";
        message.textContent = "Upload or fetch a judgment to start the AI workflow.";
        bar.style.width = "0%";
        stageList.innerHTML = "";
        return;
    }

    const state = payload.state || "running";
    container.className = `task-status ${state === "success" ? "success" : state === "failure" ? "failure" : "running"}`;
    title.textContent = state === "success" ? "Ready for human review" : state === "failure" ? "AI workflow failed" : "AI agent is processing";
    message.textContent = payload.error || payload.message || humanize(payload.stage || "Processing judgment");
    bar.style.width = `${Math.max(0, Math.min(100, Number(payload.pct || 0)))}%`;
    stageList.innerHTML = (payload.stages || [])
        .map((stage) => `<div class="task-stage ${escapeHtml(stage.state || "pending")}">${escapeHtml(stage.label || humanize(stage.key))}</div>`)
        .join("");
}

function renderReviewSkeleton(recordId) {
    const pane = document.getElementById("reviewPane");
    pane.classList.remove("closing");
    pane.classList.add("open", "loading");
    document.getElementById("reviewTitle").textContent = "Loading review package";
    document.getElementById("reviewMeta").textContent = recordId ? `Record ${recordId.slice(0, 8)}...` : "Fetching evidence, extraction, and action plan";
    document.getElementById("recordWarnings").innerHTML = "";
    document.getElementById("duplicatePanel").innerHTML = "";
    document.getElementById("actionsContainer").innerHTML = `<div class="review-skeleton"><span></span><span></span><span></span></div>`;
    document.getElementById("fieldsContainer").innerHTML = `<div class="review-skeleton"><span></span><span></span><span></span></div>`;
    document.getElementById("recordMetrics").innerHTML = "";
    document.getElementById("auditTimeline").innerHTML = "";
    document.getElementById("pdfFrame").removeAttribute("src");
    currentPdfPage = 1;
    currentPdfPageCount = 1;
    currentPdfRecordId = null;
    updatePdfViewerControls();
}

function closeReviewPane() {
    const pane = document.getElementById("reviewPane");
    pane.classList.add("closing");
    window.setTimeout(() => {
        pane.classList.remove("open", "loading", "closing");
    }, window.matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : 420);
}

function formatList(values, fallback = "None") {
    const text = formatValue(values);
    return text || fallback;
}

function normalizeDisplayList(values) {
    if (Array.isArray(values)) {
        return values.map((value) => formatValue(value).trim()).filter(Boolean);
    }
    if (typeof values === "string") {
        return values.split(/[,;\n]+/).map((value) => value.trim()).filter(Boolean);
    }
    const value = formatValue(values).trim();
    return value ? [value] : [];
}

function formatInlineList(values, fallback = "None") {
    const items = normalizeDisplayList(values);
    return items.length ? items.join(", ") : fallback;
}

function riskSeverity(flag) {
    const text = String(flag || "").toLowerCase();
    if (!text || text === "none") return "low";
    if (text === "llm_case_details_failed") return "medium";
    if (/(missing|unsupported|failed|error|critical|overdue|no source|owner unclear|court|deadline|department)/.test(text)) {
        return "high";
    }
    if (/(possible|duplicate|ambigu|ocr|discarded|low confidence|review)/.test(text)) {
        return "medium";
    }
    return "low";
}

function countMissingFields(record = {}) {
    if (Number.isFinite(Number(record.missing_field_count))) {
        return Number(record.missing_field_count);
    }
    const extraction = record.extraction || {};
    return Object.entries(extraction)
        .filter(([key, field]) => {
            if (["directions", "risk_flags", "legal_phrases"].includes(key) || !field || typeof field !== "object") return false;
            const value = field.value;
            return value === null || value === undefined || value === "" || (Array.isArray(value) && !value.length) || String(value).trim().toLowerCase() === "unknown";
        }).length;
}

function formatRiskFlag(flag, record = {}) {
    if (String(flag || "").toLowerCase() === "llm_case_details_failed") {
        const count = countMissingFields(record);
        if (!count) return "Fields not filled";
        return `${count} ${count === 1 ? "field" : "fields"} not filled`;
    }
    return humanize(flag);
}

function renderRiskFlags(flags = [], record = {}) {
    const normalized = flags.map((flag) => formatRiskFlag(flag, record)).filter(Boolean);
    if (!normalized.length) {
        return `<span class="risk-chip risk-low">Clear</span>`;
    }
    return flags.map((rawFlag, index) => {
        const label = normalized[index] || formatRiskFlag(rawFlag, record);
        const severity = riskSeverity(rawFlag);
        return `<span class="risk-chip risk-${severity}">${escapeHtml(label)}</span>`;
    }).join("");
}

function formatCitation(evidence) {
    if (!evidence) return "Needs source";
    const page = evidence.page || evidence.page_start || evidence.pages?.[0] || "?";
    const method = [
        evidence.source_quality,
        evidence.source_type,
        evidence.extractor_name,
        evidence.extraction_method,
    ].map(humanize).find((label) => label && !/^(Llm|Llm First|Llm Cited Snippet)$/i.test(label));
    return method ? `p. ${page} | ${method}` : `p. ${page}`;
}

function renderEvidenceList(evidenceItems = [], emptyText = "No source found") {
    const items = (evidenceItems || []).filter(Boolean).slice(0, 4);
    if (!items.length) {
        return `<div class="source-chip-list missing-evidence"><span class="source-chip source-missing">${escapeHtml(emptyText)}</span></div>`;
    }
    return `
        <div class="source-chip-list">
            ${items.map((evidence, index) => {
                const snippet = evidence.snippet || "";
                const title = snippet ? ` title="${escapeHtml(snippet)}"` : "";
                return `<span class="source-chip"${title}>${escapeHtml(index + 1)}. ${escapeHtml(formatCitation(evidence))}</span>`;
            }).join("")}
        </div>
    `;
}

function renderEvidenceQuotes(evidenceItems = [], emptyText = "No source found") {
    const items = (evidenceItems || []).filter(Boolean).slice(0, 3);
    if (!items.length) return `<blockquote>${escapeHtml(emptyText)}</blockquote>`;
    return items.map((evidence, index) => `
        <blockquote>
            <small>${escapeHtml(index + 1)}. ${escapeHtml(formatCitation(evidence))}</small>
            ${escapeHtml(evidence.snippet || "Source text unavailable")}
        </blockquote>
    `).join("");
}

function reviewReasonOptions(selected = "") {
    const options = [
        ["", "Reason"],
        ["source_verified", "Source verified"],
        ["ai_value_wrong", "AI value wrong"],
        ["missing_or_empty", "Missing / empty"],
        ["ambiguous_source", "Ambiguous source"],
        ["format_cleanup", "Format cleanup"],
        ["manual_correction", "Manual correction"],
    ];
    const hasSelected = options.some(([value]) => value === selected);
    const merged = hasSelected || !selected ? options : [...options, [selected, humanize(selected)]];
    return merged.map(([value, label]) => `<option value="${escapeHtml(value)}" ${value === selected ? "selected" : ""}>${escapeHtml(label)}</option>`).join("");
}

function renderKpis(records, queueRecords = []) {
    const publishedActions = records.reduce((count, record) => count + (record.pending_actions || []).length, 0);
    const pendingReview = queueRecords.filter((record) => record.review_status === "pending_review").length;
    const escalated = queueRecords.filter((record) => ["needs_clarification", "escalated"].includes(record.review_status)).length;
    const approvedToday = records.filter((record) => record.review_status === "approved").length;
    const html = [
        { label: "Published actions", value: publishedActions, icon: "/frontend/assets/kpi-document.png" },
        { label: "Pending review", value: pendingReview, icon: "/frontend/assets/kpi-check.png" },
        { label: "Escalated", value: escalated, icon: "/frontend/assets/kpi-alert.png" },
        { label: "Approved today", value: approvedToday, icon: "/frontend/assets/kpi-shield-purple.png" },
    ].map((item) => `
        <div class="kpi">
            <span class="kpi-icon"><img src="${escapeHtml(item.icon)}" alt=""></span>
            <strong>${item.value}</strong>
            <span>${item.label}</span>
        </div>
    `).join("");
    document.getElementById("kpiStrip").innerHTML = html;
}

function renderMetrics(containerId, metrics = {}) {
    const items = [
        ["Evidence coverage", `${metrics.evidence_coverage_percent ?? 0}%`],
        ["Duplicates", metrics.duplicate_count ?? 0],
        ["Ambiguity", metrics.ambiguous_count ?? 0],
        ["Reviewer edits", metrics.review_edit_count ?? 0],
        ["OCR", metrics.ocr_used ? (metrics.vision_ocr_used ? "Vision OCR" : "Used") : "Not used"],
    ];
    document.getElementById(containerId).innerHTML = items.map(([label, value]) => `
        <div class="metric-card"><strong>${escapeHtml(value)}</strong><span>${escapeHtml(label)}</span></div>
    `).join("");
}

function renderQueue(records) {
    const body = document.getElementById("queueTableBody");
    const queueCount = document.getElementById("queueCount");
    if (queueCount) {
        queueCount.textContent = records.length ? `Showing 1-${records.length} of ${records.length} items` : "Showing 0 items";
    }
    if (!records.length) {
        body.innerHTML = `<tr><td colspan="6">No records in this queue.</td></tr>`;
        return;
    }
    body.innerHTML = records.map((record) => `
        <tr>
            <td>${escapeHtml(record.case_number || record.record_id)}</td>
            <td>${escapeHtml(record.court || "Unknown court")}</td>
            <td>${escapeHtml(formatInlineList(record.departments, "Owner unclear"))}</td>
            <td>${statusPill(record.review_status || "pending_review")}</td>
            <td><div class="risk-stack">${renderRiskFlags(record.risk_flags || [], record)}</div></td>
            <td>
                <div class="queue-actions">
                    <button class="ghost-btn" data-record-id="${escapeHtml(record.record_id)}">Review</button>
                    <button class="remove-review-btn" data-remove-record-id="${escapeHtml(record.record_id)}" aria-label="Remove review">Remove</button>
                </div>
            </td>
        </tr>
    `).join("");
    body.querySelectorAll("button[data-record-id]").forEach((button) => {
        button.addEventListener("click", () => loadRecord(button.dataset.recordId));
    });
    body.querySelectorAll("button[data-remove-record-id]").forEach((button) => {
        button.addEventListener("click", () => removeRecord(button.dataset.removeRecordId));
    });
}

function normalizeActionRegister(record) {
    const rows = Array.isArray(record.action_register) ? record.action_register.filter(Boolean) : [];
    if (rows.length) {
        return rows.map((row) => ({ ...row }));
    }
    const actions = normalizeDisplayList(record.pending_actions);
    const dueDates = normalizeDisplayList(record.due_dates);
    const departments = normalizeDisplayList(record.departments);
    const categories = normalizeDisplayList(record.action_categories);
    return (actions.length ? actions : ["No published action"]).map((title, index) => ({
        title,
        owner: departments[index] || departments[0] || "",
        department: departments[index] || departments[0] || "",
        category: categories[index] || categories[0] || "",
        deadline: dueDates[index] || dueDates[0] || "",
        timeline_type: dueDates[index] || dueDates[0] ? "specified" : "missing",
        status: record.review_status || "approved",
        evidence_count: 0,
    }));
}

function formatDeadlineChip(action) {
    const deadline = formatValue(action.deadline).trim();
    const timeline = formatValue(action.timeline).trim();
    if (deadline) {
        return `<span class="deadline-chip deadline-dated">${escapeHtml(deadline)}</span>`;
    }
    if (timeline && !/not specified|no timeline|none/i.test(timeline)) {
        return `<span class="deadline-chip deadline-text">${escapeHtml(timeline)}</span>`;
    }
    return `<span class="deadline-chip deadline-missing">Timeline not specified</span>`;
}

function formatEvidenceButton(record, action) {
    const evidenceCount = Number(action.evidence_count || 0);
    const page = action.evidence_page ? `p. ${action.evidence_page}` : "";
    const label = evidenceCount ? `Source${page ? ` ${page}` : ""}` : "Open proof";
    return `<button class="ghost-btn evidence-open-btn" data-record-id="${escapeHtml(record.record_id)}">${escapeHtml(label)}</button>`;
}

function renderDashboard(records) {
    const body = document.getElementById("recordsTableBody");
    if (!records.length) {
        body.innerHTML = `<tr><td colspan="7">No verified records yet. Approve a judgment to publish it here.</td></tr>`;
        return;
    }
    body.innerHTML = records.flatMap((record) => {
        const actions = normalizeActionRegister(record);
        return actions.map((action) => {
            const owner = action.owner || action.department || formatList(record.departments, "Owner unclear");
            const category = action.category ? `<span class="register-meta">${escapeHtml(humanize(action.category))}</span>` : "";
            const rowStatus = record.review_status || action.status || "approved";
            const judgmentDate = formatValue(record.judgment_date);
            return `
                <tr>
                    <td>
                        <strong class="register-case">${escapeHtml(record.case_number || "Unknown case")}</strong>
                        ${judgmentDate ? `<span class="register-meta">${escapeHtml(judgmentDate)}</span>` : ""}
                    </td>
                    <td>${escapeHtml(record.court || "Unknown court")}</td>
                    <td>
                        <strong>${escapeHtml(formatValue(owner) || "Owner unclear")}</strong>
                        ${category}
                    </td>
                    <td>${escapeHtml(formatValue(action.title) || "No action captured")}</td>
                    <td>${formatDeadlineChip(action)}</td>
                    <td>${statusPill(rowStatus)}</td>
                    <td>${formatEvidenceButton(record, action)}</td>
                </tr>
            `;
        });
    }).join("");
    body.querySelectorAll("button[data-record-id]").forEach((button) => {
        button.addEventListener("click", () => loadRecord(button.dataset.recordId));
    });
}

async function refreshAll() {
    const userId = getUserId();
    const queueParams = new URLSearchParams();
    if (activeQueueStatus) queueParams.set("status", activeQueueStatus);
    const dashboardParams = new URLSearchParams();
    const caseFilter = document.getElementById("filterCase").value.trim();
    const departmentFilter = document.getElementById("filterDepartment").value.trim();
    const actionType = document.getElementById("filterActionType").value;
    if (caseFilter) dashboardParams.set("case_query", caseFilter);
    if (departmentFilter) dashboardParams.set("department", departmentFilter);
    if (actionType) dashboardParams.set("action_type", actionType);

    const [queuePayload, dashboardPayload] = await Promise.all([
        apiRequest(`/judgments/records/${userId}?${queueParams.toString()}`),
        apiRequest(`/judgments/dashboard/${userId}?${dashboardParams.toString()}`),
    ]);
    const queueRecords = queuePayload.records || [];
    const dashboardRecords = dashboardPayload.records || [];
    renderQueue(queueRecords);
    renderDashboard(dashboardRecords);
    renderKpis(dashboardRecords, queueRecords);
    renderMetrics("metricsStrip", dashboardPayload.metrics || {});
}

function renderField(fieldKey, field) {
    const evidenceItems = field.evidence || [];
    const evidence = evidenceItems[0];
    const evidenceText = evidence ? `p. ${evidence.page || "?"} - ${evidence.snippet || ""}` : "No source found";
    const citation = formatCitation(evidence);
    const value = formatFieldDisplayValue(fieldKey, field.value);
    return `
        <div class="field-card" data-field-key="${escapeHtml(fieldKey)}">
            <div class="card-top">
                <label>${escapeHtml(humanize(fieldKey))}</label>
            </div>
            <label class="editable-fill field-value-fill">
                <input data-field-value value="${escapeHtml(value)}">
                <span data-edit-icon aria-hidden="true">✎</span>
            </label>
            <div class="field-original">AI proposed: ${escapeHtml(formatValue(field.ai_value ?? field.raw_value ?? field.value) || "Not available")}</div>
            <div class="mini-row">
                <select data-field-status>
                    ${statusOptions(field.status || "pending_review", ["pending_review", "approved", "edited", "ambiguous", "rejected", "empty"], { rejected: "Wrong", empty: "Empty" })}
                </select>
                <select data-field-reason>${reviewReasonOptions(field.reason || "")}</select>
            </div>
            <label class="manual-override-row">
                <input type="checkbox" data-field-manual-override ${field.manual_override ? "checked" : ""}>
                <span>Manual override</span>
            </label>
            <div class="evidence ${evidence ? "" : "missing-evidence"}">${escapeHtml(evidenceText)}</div>
            <div class="field-provenance ${evidence ? "" : "missing-evidence"}">${escapeHtml(citation)}</div>
            ${renderEvidenceList(evidenceItems)}
            <div class="confidence">Confidence: ${escapeHtml(field.confidence ?? 0)}</div>
        </div>
    `;
}

function renderAction(action, index = 0) {
    const evidenceItems = action.evidence || [];
    const evidence = evidenceItems[0];
    const evidenceText = evidence ? `p. ${evidence.page || "?"} - ${evidence.snippet || ""}` : "No source found";
    const deadline = (action.timeline || {}).raw_text || "not explicitly mentioned";
    const flags = formatList((action.ambiguity_flags || []).map(humanize), "None");
    return `
        <div class="action-card" data-action-id="${escapeHtml(action.action_id)}">
            <div class="action-card-main">
                <div class="action-card-title-row">
                    <span class="action-index">${index + 1}</span>
                    <div>
                        <label class="editable-fill action-title-fill">
                            <input class="action-title-input" data-prop="title" value="${escapeHtml(action.title || "")}" aria-label="Action title">
                            <span data-edit-icon aria-hidden="true">✎</span>
                        </label>
                        <p>${escapeHtml(humanize(action.category || "compliance"))}</p>
                    </div>
                    <span class="action-card-toggle" aria-hidden="true">⌃</span>
                </div>
                <input type="hidden" data-prop="category" value="${escapeHtml(action.category || "compliance")}">
                <div class="action-card-controls">
                    <label>
                        <span>Owner</span>
                        <span class="editable-fill">
                            <input data-prop="responsible_department" placeholder="Owner" value="${escapeHtml(action.responsible_department || "")}">
                            <span data-edit-icon aria-hidden="true">✎</span>
                        </span>
                    </label>
                    <label>
                        <span>Priority</span>
                        <select data-prop="priority">${statusOptions(action.priority || "medium", ["low", "medium", "high", "critical"])}</select>
                    </label>
                    <label>
                        <span>Status</span>
                        <select data-prop="status">${statusOptions(action.status || "pending_review", ["pending_review", "approved", "edited", "rejected", "escalated", "completed"])}</select>
                    </label>
                </div>
                <div class="action-meta-strip">
                    <div>
                        <svg class="button-icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M7 3v3M17 3v3M4 9h16M6 5h12a2 2 0 0 1 2 2v11a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2Z"/></svg>
                        <span>Deadline</span>
                        <strong>${escapeHtml(humanize(deadline))}</strong>
                    </div>
                    <div>
                        <svg class="button-icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M5 21V4h10l1 3h4v9h-9l-1-3H5"/></svg>
                        <span>Flags</span>
                        <strong>${escapeHtml(flags)}</strong>
                    </div>
                </div>
                <label class="source-proof">
                    <span>Source proof</span>
                    ${renderEvidenceList(evidenceItems)}
                    ${renderEvidenceQuotes(evidenceItems)}
                    <span class="source-proof-note">
                        <span>Reviewer note</span>
                        <span class="editable-fill">
                            <input data-prop="reason" placeholder="Add reviewer note or reason" value="${escapeHtml(action.reason || "")}">
                            <span data-edit-icon aria-hidden="true">✎</span>
                        </span>
                    </span>
                    <span class="source-proof-note">
                        <label class="manual-override-row">
                            <input type="checkbox" data-prop="manual_override" ${action.manual_override ? "checked" : ""}>
                            <span>Manual override</span>
                        </label>
                    </span>
                </label>
            </div>
        </div>
    `;
}

function statusOptions(selected, options, labels = {}) {
    return options.map((option) => `<option value="${option}" ${option === selected ? "selected" : ""}>${escapeHtml(labels[option] || humanize(option))}</option>`).join("");
}

function categoryOptions(selected) {
    return ["compliance", "appeal_consideration", "legal_review", "information_update", "affidavit_report_filing", "payment_release", "reconsideration", "no_immediate_action"]
        .map((option) => `<option value="${option}" ${option === selected ? "selected" : ""}>${humanize(option)}</option>`).join("");
}

function renderWarnings(record) {
    const flags = [...(record.risk_flags || []), ...((record.pdf_profile || {}).profile_type !== "digital" ? [`PDF profile: ${(record.pdf_profile || {}).profile_type}`] : [])];
    if (record.duplicate_candidates && record.duplicate_candidates.length) {
        flags.push("possible duplicate");
    }
    document.getElementById("recordWarnings").innerHTML = flags.map((flag) => `<span>${escapeHtml(formatRiskFlag(flag, record))}</span>`).join("");
}

async function loadRecord(recordId) {
    renderReviewSkeleton(recordId);
    const userId = getUserId();
    const record = await apiRequest(`/judgments/${recordId}?user_id=${encodeURIComponent(userId)}`);
    currentRecord = record;
    document.getElementById("reviewPane").classList.remove("loading");
    document.getElementById("reviewTitle").textContent = (record.extraction.case_number || {}).value || record.record_id;
    document.getElementById("reviewMeta").textContent = `${(record.extraction.court || {}).value || "Unknown court"} | ${humanize(record.review_status || "pending_review")}`;
    renderWarnings(record);
    document.getElementById("fieldsContainer").innerHTML = Object.entries(record.extraction)
        .filter(([key, value]) => !["directions", "risk_flags", "legal_phrases"].includes(key) && value && typeof value === "object" && "value" in value)
        .map(([key, field]) => renderField(key, field))
        .join("");
    document.getElementById("actionsContainer").innerHTML = (record.action_items || []).map(renderAction).join("");
    await Promise.all([fetchHighlightedPdf(recordId), loadAudit(recordId), loadRecordMetrics(recordId), loadDuplicates(recordId)]);
}

function highlightedPdfUrl(recordId, userId) {
    return `/judgments/${encodeURIComponent(recordId)}/highlighted-pdf?user_id=${encodeURIComponent(userId)}`;
}

function highlightedPageUrl(recordId, userId, pageNumber = 1) {
    return `/judgments/${encodeURIComponent(recordId)}/highlighted-page/${encodeURIComponent(pageNumber)}?user_id=${encodeURIComponent(userId)}`;
}

function updatePdfViewerControls() {
    const input = document.getElementById("pdfPageInput");
    const total = document.getElementById("pdfPageTotal");
    const prev = document.getElementById("pdfPrevBtn");
    const next = document.getElementById("pdfNextBtn");
    const label = document.getElementById("pdfPageLabel");
    if (!input || !total || !prev || !next || !label) return;

    input.min = "1";
    input.max = String(currentPdfPageCount);
    input.value = String(currentPdfPage);
    total.textContent = `of ${currentPdfPageCount}`;
    label.textContent = "Page";
    prev.disabled = currentPdfPage <= 1;
    next.disabled = currentPdfPage >= currentPdfPageCount;
}

function loadHighlightedPage(pageNumber) {
    if (!currentPdfRecordId) return;
    const boundedPage = Math.min(Math.max(Number(pageNumber) || 1, 1), currentPdfPageCount);
    currentPdfPage = boundedPage;
    const userId = getUserId();
    document.getElementById("pdfFrame").src = `${apiBase}${highlightedPageUrl(currentPdfRecordId, userId, boundedPage)}`;
    updatePdfViewerControls();
}

async function fetchHighlightedPdf(recordId) {
    const userId = getUserId();
    if (currentPdfUrl) {
        currentPdfUrl = null;
    }
    const pageCount = Number(((currentRecord || {}).pdf_profile || {}).page_count) || 1;
    currentPdfRecordId = recordId;
    currentPdfPageCount = Math.max(pageCount, 1);
    currentPdfPage = 1;
    currentPdfUrl = `${apiBase}${highlightedPdfUrl(recordId, userId)}`;
    document.getElementById("pdfLink").href = currentPdfUrl;
    document.getElementById("pdfFrame").src = `${apiBase}${highlightedPageUrl(recordId, userId, 1)}`;
    updatePdfViewerControls();
}

function wirePdfViewerControls() {
    const prev = document.getElementById("pdfPrevBtn");
    const next = document.getElementById("pdfNextBtn");
    const input = document.getElementById("pdfPageInput");
    if (!prev || !next || !input) return;

    prev.addEventListener("click", () => loadHighlightedPage(currentPdfPage - 1));
    next.addEventListener("click", () => loadHighlightedPage(currentPdfPage + 1));
    input.addEventListener("change", () => loadHighlightedPage(input.value));
    input.addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
            event.preventDefault();
            loadHighlightedPage(input.value);
        }
    });
    updatePdfViewerControls();
}

async function loadAudit(recordId) {
    const userId = getUserId();
    const payload = await apiRequest(`/judgments/${recordId}/audit?user_id=${encodeURIComponent(userId)}`);
    const events = payload.events || [];
    document.getElementById("auditTimeline").innerHTML = events.length ? events.map((event) => `
        <div class="audit-event">
            <strong>${escapeHtml(humanizeAuditLabel(event.event_type))}</strong>
            <span>${escapeHtml(event.created_at || "")}</span>
            <p>${escapeHtml(formatAuditValue(event.notes || event.reason || ""))}</p>
            ${event.before || event.after ? `<p>${escapeHtml(auditDiff(event.before, event.after))}</p>` : ""}
        </div>
    `).join("") : "No audit events yet.";
}

function auditDiff(before, after) {
    const beforeValue = auditSummaryValue(before);
    const afterValue = auditSummaryValue(after);
    return `Before: ${formatAuditValue(beforeValue) || "None"} | After: ${formatAuditValue(afterValue) || "None"}`;
}

async function loadRecordMetrics(recordId) {
    const userId = getUserId();
    const payload = await apiRequest(`/judgments/${recordId}/metrics?user_id=${encodeURIComponent(userId)}`);
    renderMetrics("recordMetrics", payload.metrics || {});
}

async function loadDuplicates(recordId) {
    const userId = getUserId();
    const payload = await apiRequest(`/judgments/${recordId}/duplicates?user_id=${encodeURIComponent(userId)}`);
    const candidates = payload.candidates || currentRecord.duplicate_candidates || [];
    const resolution = payload.resolution;
    const panel = document.getElementById("duplicatePanel");
    if (resolution) {
        const duplicateLabel = await getDuplicateLabel(resolution.duplicate_record_id);
        panel.innerHTML = `<strong>Duplicate resolved:</strong> ${escapeHtml(humanize(resolution.resolution))}${duplicateLabel ? ` against <span>${escapeHtml(duplicateLabel)}</span>` : ""}`;
        return;
    }
    if (!candidates.length) {
        panel.innerHTML = "";
        return;
    }
    panel.innerHTML = `
        <strong>Possible duplicate found</strong>
        ${candidates.map((candidate) => `
            <div class="duplicate-row">
                <span>${escapeHtml(candidate.original_file_name || candidate.case_number || candidate.record_id)} (${escapeHtml(candidate.duplicate_score || "hash match")})</span>
                <span>
                    <button class="ghost-btn" data-open-duplicate="${escapeHtml(candidate.record_id)}">Open</button>
                    <button class="secondary-btn" data-resolve-duplicate="${escapeHtml(candidate.record_id)}">Use Existing</button>
                </span>
            </div>
        `).join("")}
        <div class="duplicate-row">
            <span>This is not a duplicate</span>
            <button class="ghost-btn" data-keep-both="true">Keep Both</button>
        </div>
    `;
    panel.querySelectorAll("[data-open-duplicate]").forEach((button) => {
        button.addEventListener("click", () => loadRecord(button.dataset.openDuplicate));
    });
    panel.querySelectorAll("[data-resolve-duplicate]").forEach((button) => {
        button.addEventListener("click", () => resolveDuplicate("use_existing", button.dataset.resolveDuplicate));
    });
    const keepBoth = panel.querySelector("[data-keep-both]");
    if (keepBoth) keepBoth.addEventListener("click", () => resolveDuplicate("keep_both", null));
}

async function getDuplicateLabel(recordId) {
    if (!recordId) return "";
    const matchingCandidate = (currentRecord?.duplicate_candidates || []).find((candidate) => candidate.record_id === recordId);
    if (matchingCandidate) {
        return matchingCandidate.original_file_name || matchingCandidate.case_number || matchingCandidate.record_id;
    }
    try {
        const record = await apiRequest(`/judgments/${recordId}?user_id=${encodeURIComponent(getUserId())}`);
        return (record.source_metadata || {}).original_file_name
            || (record.extraction?.case_number || {}).value
            || record.record_id
            || recordId;
    } catch (_) {
        return recordId;
    }
}

async function resolveDuplicate(resolution, duplicateRecordId) {
    if (!currentRecord) return;
    const userId = getUserId();
    await apiRequest(`/judgments/${currentRecord.record_id}/duplicates/resolve?user_id=${encodeURIComponent(userId)}`, {
        method: "POST",
        body: JSON.stringify({
            reviewer_id: getUserId(),
            resolution,
            duplicate_record_id: duplicateRecordId,
            notes: document.getElementById("reviewNotes").value,
        }),
    });
    showToast("Duplicate resolution saved.");
    await loadDuplicates(currentRecord.record_id);
    await loadAudit(currentRecord.record_id);
}

function collectReviewPayload(decision) {
    const extractionUpdates = {};
    document.querySelectorAll(".field-card").forEach((card) => {
        const key = card.dataset.fieldKey;
        extractionUpdates[key] = {
            field_id: key,
            value: card.querySelector("[data-field-value]").value,
            status: card.querySelector("[data-field-status]").value,
            reason: card.querySelector("[data-field-reason]").value,
            manual_override: card.querySelector("[data-field-manual-override]").checked,
        };
    });
    const actionUpdates = Array.from(document.querySelectorAll(".action-card")).map((card) => {
        const update = { action_id: card.dataset.actionId };
        card.querySelectorAll("[data-prop]").forEach((input) => {
            update[input.dataset.prop] = input.type === "checkbox" ? input.checked : input.value;
        });
        if (decision === "reject") update.status = "rejected";
        if (decision === "approve" && !update.status) update.status = "approved";
        return update;
    });
    return {
        reviewer_id: getUserId(),
        reviewer_role: getReviewerRole(),
        decision,
        notes: document.getElementById("reviewNotes").value,
        extraction_updates: extractionUpdates,
        action_updates: actionUpdates,
    };
}

async function submitReview(decision) {
    if (!currentRecord) return;
    const userId = getUserId();
    const payload = collectReviewPayload(decision);
    await apiRequest(`/judgments/${currentRecord.record_id}/review?user_id=${encodeURIComponent(userId)}`, {
        method: "POST",
        body: JSON.stringify(payload),
    });
    showToast(`Record ${decision} submitted.`);
    await refreshAll();
    await loadRecord(currentRecord.record_id);
}

async function removeRecord(recordId) {
    const confirmed = window.confirm("Remove this review from the queue? This only clears it from the prototype workspace.");
    if (!confirmed) return;
    const userId = getUserId();
    await apiRequest(`/judgments/${recordId}?user_id=${encodeURIComponent(userId)}`, { method: "DELETE" });
    if (currentRecord && currentRecord.record_id === recordId) {
        closeReviewPane();
        currentRecord = null;
    }
    showToast("Review removed from queue.");
    await refreshAll();
}

async function queueUpload(file) {
    const userId = getUserId();
    const formData = new FormData();
    formData.append("user_id", userId);
    formData.append("file", file);
    const payload = await apiRequest("/judgments/upload-progress", { method: "POST", body: formData });
    await handleProgressPayload(payload, "Judgment uploaded.");
}

async function queueCcmsFetch(ccmsCaseId) {
    const payload = await apiRequest("/judgments/from-ccms-progress", {
        method: "POST",
        body: JSON.stringify({ user_id: getUserId(), ccms_case_id: ccmsCaseId }),
    });
    await handleProgressPayload(payload, "CCMS fetch queued.");
}

async function loadDemoSeed() {
    const payload = await apiRequest(`/judgments/demo-seed-progress?user_id=${encodeURIComponent(getUserId())}`, { method: "POST" });
    await handleProgressPayload(payload, "Demo judgment loaded.");
}

async function handleProgressPayload(payload, message) {
    showToast(message);
    activeProgressJob = payload.job_id;
    setTaskProgress(payload);
    await pollProgressJob(payload.job_id);
}

async function handleQueuedPayload(payload, message) {
    showToast(message);
    if (payload.state === "SUCCESS") {
        setTaskProgress({ state: "success", pct: 100, message: "Judgment processing complete.", stages: [] });
        await refreshAll();
        await loadRecord(payload.record_id);
        return;
    }
    await pollTask(payload.task_id);
}

async function pollTask(taskId) {
    setTaskProgress({ state: "running", pct: 10, message: "Processing judgment...", stages: [] });
    for (let attempt = 0; attempt < 90; attempt += 1) {
        const payload = await apiRequest(`/tasks/${taskId}`);
        if (payload.state === "SUCCESS") {
            setTaskProgress({ state: "success", pct: 100, message: "Judgment processing complete.", stages: [] });
            await refreshAll();
            if (payload.result && payload.result.record_id) await loadRecord(payload.result.record_id);
            return;
        }
        if (payload.state === "FAILURE") {
            setTaskProgress({ state: "failure", pct: 100, error: payload.error || "unknown error", stages: [] });
            throw new Error(payload.error || "Judgment task failed");
        }
        setTaskProgress({ state: "running", pct: 10, message: payload.status || `Task state: ${payload.state}`, stages: [] });
        await new Promise((resolve) => window.setTimeout(resolve, 2000));
    }
    setTaskProgress({ state: "failure", pct: 100, error: "Task timed out while polling.", stages: [] });
}

async function pollProgressJob(jobId) {
    for (let attempt = 0; attempt < 900; attempt += 1) {
        const payload = await apiRequest(`/judgments/jobs/${jobId}`);
        if (activeProgressJob !== jobId) return;
        setTaskProgress(payload);
        if (payload.state === "success") {
            await refreshAll();
            await loadRecord((payload.result && payload.result.record_id) || payload.record_id);
            return;
        }
        if (payload.state === "failure") {
            throw new Error(payload.error || "Judgment processing failed");
        }
        await new Promise((resolve) => window.setTimeout(resolve, 900));
    }
    setTaskProgress({ state: "failure", pct: 100, error: "AI workflow timed out while polling.", stages: [] });
}

async function exportCsv() {
    const response = await apiRequest(`/judgments/dashboard/${getUserId()}/export?format=csv`);
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "judgment_dashboard.csv";
    link.click();
    URL.revokeObjectURL(url);
}

document.getElementById("uploadForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    const file = document.getElementById("pdfInput").files[0];
    if (!file) return showToast("Choose a PDF first.");
    try { await queueUpload(file); } catch (error) { showToast(error.message); }
});

document.getElementById("ccmsForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    const ccmsCaseId = document.getElementById("ccmsCaseId").value.trim();
    if (!ccmsCaseId) return showToast("Enter a CCMS case ID.");
    try { await queueCcmsFetch(ccmsCaseId); } catch (error) { showToast(error.message); }
});

document.getElementById("demoSeedBtn").addEventListener("click", () => loadDemoSeed().catch((error) => showToast(error.message)));
document.getElementById("refreshBtn").addEventListener("click", () => refreshAll().catch((error) => showToast(error.message)));
document.getElementById("exportCsvBtn").addEventListener("click", () => exportCsv().catch((error) => showToast(error.message)));
document.getElementById("pdfInput").addEventListener("change", (event) => {
    document.getElementById("pdfFileName").textContent = event.target.files?.[0]?.name || "No judgment selected";
});
document.getElementById("filterCase").addEventListener("input", () => refreshAll().catch((error) => showToast(error.message)));
document.getElementById("filterDepartment").addEventListener("input", () => refreshAll().catch((error) => showToast(error.message)));
document.getElementById("filterActionType").addEventListener("change", () => refreshAll().catch((error) => showToast(error.message)));
document.getElementById("queueTabs").querySelectorAll("button[data-status]").forEach((button) => {
    button.addEventListener("click", () => {
        document.querySelectorAll("#queueTabs button").forEach((item) => item.classList.remove("active"));
        button.classList.add("active");
        activeQueueStatus = button.dataset.status || "";
        refreshAll().catch((error) => showToast(error.message));
    });
});
document.getElementById("closeReviewBtn").addEventListener("click", closeReviewPane);
document.getElementById("approveBtn").addEventListener("click", () => submitReview("approve").catch((error) => showToast(error.message)));
document.getElementById("editBtn").addEventListener("click", () => {
    const firstEditable = document.querySelector("#fieldsContainer [data-field-value], #actionsContainer [data-prop]");
    if (firstEditable) firstEditable.focus();
    showToast("Extraction fields are editable inline. Approve to publish the reviewed values.");
});
document.getElementById("rejectBtn").addEventListener("click", () => submitReview("reject").catch((error) => showToast(error.message)));
document.getElementById("escalateBtn").addEventListener("click", () => submitReview("escalate").catch((error) => showToast(error.message)));
document.getElementById("completeBtn").addEventListener("click", () => submitReview("complete").catch((error) => showToast(error.message)));

wirePdfViewerControls();
setIdentityBanner();
refreshAll().catch((error) => showToast(error.message));
