import { postFormData, getJSON, postJSON } from '../api.js';
import { doSwitchView } from '../app.js';

window.openMonitor = openMonitor;
window.restartFromTask = restartFromTask;
window.startNewProject = startNewProject;

let currentWs = null;
let currentTaskId = null;
let downloadedVideoPath = null;

// Speaker merge state
let speakerMergeState = {
    selected: new Set(),
    groups: [] // Array of arrays: [[0, 2], [1]] means 0&2 merged, 1 separate
};

export function init() {
    // 1. Bind Upload Form
    const uploadForm = document.getElementById('upload-form');
    if (uploadForm) uploadForm.addEventListener('submit', handleUpload);
    
    // 2. Bind YouTube Download
    const ytBtn = document.getElementById('download-youtube-btn');
    if (ytBtn) ytBtn.addEventListener('click', handleYouTubeDownload);
    
    // 3. Bind File Input Change (visual feedback)
    const fileInput = document.getElementById('video-input');
    if (fileInput) fileInput.addEventListener('change', updateFileDisplay);

    // 4. Bind Validation Confirm
    const startDubBtn = document.getElementById('start-dub-btn');
    if (startDubBtn) startDubBtn.addEventListener('click', startTranslationPhase);
    
    // 5. Bind "New Project" button on result page
    const newProjectBtn = document.getElementById('result-new-project-btn');
    if (newProjectBtn) {
        newProjectBtn.addEventListener('click', (e) => {
            e.preventDefault();
            startNewProject();
        });
    }

    // 6. Bind merge controls
    const applyMergeBtn = document.getElementById('apply-merge-btn');
    if (applyMergeBtn) {
        applyMergeBtn.addEventListener('click', applySpeakerMerge);
    }
}

export function onShow() {
    console.log('VideoDub view shown. Current task ID:', currentTaskId);
    
    // If we have an active task ID, we're in monitor mode - resume monitoring
    if (currentTaskId) {
        loadAndMonitor(currentTaskId);
    }
    // Otherwise, the HTML already shows the upload step by default (active class)
    // No need to explicitly call showWizardStep here - let the HTML state persist
}

export function startNewProject() {
    console.log('Starting fresh new project');
    
    // Reset all state
    currentTaskId = null;
    downloadedVideoPath = null;
    speakerMergeState = { selected: new Set(), groups: [] };
    
    // Close any existing WebSocket
    if (currentWs) {
        currentWs.close();
        currentWs = null;
    }
    
    // Reset the upload form
    const uploadForm = document.getElementById('upload-form');
    if (uploadForm) uploadForm.reset();
    
    // Reset file display
    const fileDisplay = document.getElementById('file-display-text');
    if (fileDisplay) fileDisplay.innerText = 'Click to browse or drag video file here';
    
    const dropZone = document.getElementById('drop-zone');
    if (dropZone) dropZone.style.borderColor = 'var(--border)';
    
    const youtubeStatus = document.getElementById('youtube-status');
    if (youtubeStatus) {
        youtubeStatus.style.display = 'none';
        youtubeStatus.innerText = '';
    }
    
    const startBtn = document.getElementById('start-btn');
    if (startBtn) startBtn.disabled = true;
    
    // Clear any previous terminal logs
    const term = document.getElementById('processing-term');
    if (term) term.innerHTML = '';
    
    // Reset granular task list
    const taskList = document.getElementById('granular-task-list');
    if (taskList) {
        taskList.innerHTML = '<div class="empty-state">Waiting for workflow...</div>';
    }
    
    // Reset progress bar
    const progressBar = document.getElementById('main-progress-bar');
    if (progressBar) progressBar.style.width = '0%';
    
    const progressText = document.getElementById('processing-percent');
    if (progressText) progressText.innerText = '0%';
    
    const statusText = document.getElementById('processing-status-text');
    if (statusText) statusText.innerText = 'Initializing...';
    
    // Reset video player
    const finalVideo = document.getElementById('final-video');
    if (finalVideo) {
        finalVideo.src = '';
        finalVideo.load();
    }
    
    // Reset merge UI
    resetMergeUI();
    
    // Show upload step
    showWizardStep('state-upload');
    
    // Switch to video dub view if not already there
    const viewEl = document.getElementById('view-video_dub');
    if (viewEl && !viewEl.classList.contains('active')) {
        doSwitchView('video_dub');
    }
}

export function openMonitor(taskId) {
    // Don't switch view if already on video_dub - just load the task
    const currentView = document.querySelector('.view-section.active');
    if (!currentView || currentView.id !== 'view-video_dub') {
        doSwitchView('video_dub');
    }
    loadAndMonitor(taskId);
}

// --- HELPER TO SWITCH WIZARD STEPS ---
function showWizardStep(stepId) {
    // Hide all steps
    document.querySelectorAll('.wizard-step').forEach(el => {
        el.classList.add('hidden');
        el.classList.remove('active');
    });
    
    // Show target step
    const target = document.getElementById(stepId);
    if (target) {
        target.classList.remove('hidden');
        target.classList.add('active');
    } else {
        console.error(`Wizard step not found: ${stepId}`);
    }
}

async function handleYouTubeDownload() {
    const urlInput = document.getElementById('youtube-url');
    const statusEl = document.getElementById('youtube-status');
    const btn = document.getElementById('download-youtube-btn');
    
    const url = urlInput.value.trim();
    if (!url) return alert("Please enter a YouTube URL");

    try {
        btn.disabled = true;
        btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Fetching...';
        statusEl.style.display = 'block';
        statusEl.innerText = "Downloading video from YouTube... please wait.";
        statusEl.style.color = 'var(--text-muted)';
        
        const res = await postJSON('/api/youtube/download', { url });
        
        if (res.success) {
            downloadedVideoPath = res.file_path;
            statusEl.innerText = `✅ Downloaded: ${res.title}`;
            statusEl.style.color = 'var(--success)';
            
            document.getElementById('start-btn').disabled = false;
            document.getElementById('file-display-text').innerText = `Using YouTube: ${res.title}`;
            document.getElementById('drop-zone').style.borderColor = 'var(--success)';
        } else {
            throw new Error(res.error || "Unknown error");
        }
    } catch (e) {
        statusEl.innerText = `❌ Error: ${e.message}`;
        statusEl.style.color = 'var(--danger)';
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-cloud-download-alt"></i> Fetch';
    }
}

function updateFileDisplay(e) {
    const file = e.target.files[0];
    if (file) {
        document.getElementById('file-display-text').innerText = `Selected: ${file.name}`;
        document.getElementById('drop-zone').style.borderColor = 'var(--accent)';
        document.getElementById('start-btn').disabled = false;
        downloadedVideoPath = null; 
        const youtubeStatus = document.getElementById('youtube-status');
        if (youtubeStatus) youtubeStatus.style.display = 'none';
    }
}

async function handleUpload(e) {
    e.preventDefault();
    const btn = document.getElementById('start-btn');
    const originalBtnContent = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Starting...';

    try {
        let endpoint = '/api/upload';
        let body;
        let headers = {};

        const formData = new FormData(e.target);
        const separateAudio = document.getElementById('separate-audio').checked;
        const srcLang = document.getElementById('src-lang').value;
        const tgtLang = document.getElementById('tgt-lang').value;
        const ttsEngine = document.getElementById('tts-engine').value;

        if (downloadedVideoPath) {
            endpoint = '/api/upload-youtube';
            headers = { 'Content-Type': 'application/json' };
            body = JSON.stringify({
                file_path: downloadedVideoPath,
                filename: "YouTube_Video",
                src_lang: srcLang,
                tgt_lang: tgtLang,
                tts_engine: ttsEngine,
                separate_audio: separateAudio
            });
        } else {
            formData.set('separate_audio', separateAudio ? 'true' : 'false');
            formData.set('src_lang', srcLang);
            formData.set('tgt_lang', tgtLang);
            formData.set('tts_engine', ttsEngine);
            body = formData;
        }

        const res = await fetch(endpoint, { method: 'POST', headers, body });
        if (!res.ok) {
            const errorData = await res.json().catch(() => ({}));
            throw new Error(errorData.detail || errorData.message || `HTTP ${res.status}`);
        }
        
        const data = await res.json();
        
        // Reset downloaded video path after successful upload
        downloadedVideoPath = null;
        
        openMonitor(data.task_id);

    } catch (err) {
        alert('Error starting project: ' + err.message);
        btn.disabled = false;
        btn.innerHTML = originalBtnContent;
    }
}

async function loadAndMonitor(taskId) {
    currentTaskId = taskId;
    showWizardStep('state-processing');
    
    try {
        const task = await getJSON(`/api/projects/${taskId}`);
        renderTaskState(task);
        connectWebSocket(taskId);
    } catch(err) {
        console.error("Failed to load task", err);
        logToTerm(`Error loading task: ${err.message}`, 'error');
    }
}

function connectWebSocket(taskId) {
    if (currentWs) {
        currentWs.close();
        currentWs = null;
    }
    
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    currentWs = new WebSocket(`${proto}//${window.location.host}/ws/${taskId}`);
    
    currentWs.onmessage = (e) => {
        try {
            const msg = JSON.parse(e.data);
            console.log('WS message:', msg.type, msg);
            
            if (msg.type === 'task_progress') {
                updateTaskProgress(msg.data);
            } else if (msg.type === 'status_update') {
                renderTaskState(msg.data);
            } else if (msg.type === 'progress') {
                updateMainProgress(msg.data);
            } else if (msg.type === 'log') {
                logToTerm(msg.data.message, msg.data.style);
            } else if (msg.type === 'validation_accepted') {
                // Transition to processing after validation
                showWizardStep('state-processing');
            }
        } catch (err) {
            console.error('Error processing WS message:', err);
        }
    };
    
    currentWs.onerror = (err) => {
        console.error('WebSocket error:', err);
        logToTerm('Connection error - will retry...', 'error');
    };
    
    currentWs.onclose = () => {
        console.log('WebSocket closed');
        // Auto-reconnect if we still have an active task and aren't completed
        if (currentTaskId && !document.getElementById('state-result')?.classList.contains('active')) {
            setTimeout(() => {
                if (currentTaskId) connectWebSocket(currentTaskId);
            }, 2000);
        }
    };
}

function renderTaskState(task) {
    console.log('Rendering task state:', task.status, task.phase);
    
    // Update workflow chain if available
    if (task.workflow_tasks) {
        renderChain(task.workflow_tasks);
    }
    
    // Determine which step to show based on status/phase
    if (task.status === 'awaiting_validation' || task.phase === 'awaiting_validation') {
        showWizardStep('state-validation');
        renderSpeakerConfig(task.speaker_config || {});
        // Also set up merge UI
        setupMergeUI(Object.keys(task.speaker_config || {}));
    } else if (task.status === 'completed' || task.phase === 'complete') {
        showWizardStep('state-result');
        const finalVideo = document.getElementById('final-video');
        const downloadBtn = document.getElementById('download-btn');
        if (finalVideo && task.output_path) {
            finalVideo.src = task.output_path;
        }
        if (downloadBtn && task.output_path) {
            downloadBtn.href = task.output_path;
        }
    } else if (task.status === 'failed' || task.status === 'error') {
        // Stay on processing page but show error
        logToTerm(`FAILED: ${task.error_message || 'Unknown error'}`, 'error');
        const statusText = document.getElementById('processing-status-text');
        if (statusText) {
            statusText.innerHTML = `<span style="color:var(--danger)">FAILED: ${task.error_message || 'Unknown error'}</span>`;
        }
    } else {
        // Processing, queued, etc. - ensure we're on processing step
        const processingStep = document.getElementById('state-processing');
        if (processingStep && processingStep.classList.contains('hidden')) {
            showWizardStep('state-processing');
        }
    }
}

function renderChain(tasks) {
    const container = document.getElementById('granular-task-list');
    if (!container) return;
    
    if (!tasks || tasks.length === 0) {
        container.innerHTML = '<div class="empty-state">No workflow tasks yet...</div>';
        return;
    }
    
    let html = '';
    tasks.forEach(t => {
        const statusClass = t.status || 'pending';
        let iconClass = 'fa-circle';
        
        if (statusClass === 'completed') { iconClass = 'fa-check'; }
        else if (statusClass === 'running') { iconClass = 'fa-spinner fa-spin'; }
        else if (statusClass === 'failed') { iconClass = 'fa-times'; }
        else if (statusClass === 'skipped') { iconClass = 'fa-forward'; }
        else if (statusClass === 'cancelled') { iconClass = 'fa-ban'; }
        
        const showRestart = (statusClass === 'completed' || statusClass === 'failed');
        
        html += `
            <div id="task-node-${t.task_name}" class="task-node ${statusClass}">
                <div class="node-icon-wrapper">
                    <i class="node-icon fas ${iconClass}"></i>
                </div>
                <div class="node-content">
                    <div>
                        <span class="node-title">${t.task_name.replace(/_/g, ' ')}</span>
                        ${t.message ? `<span class="node-status-text">- ${t.message}</span>` : ''}
                    </div>
                    ${showRestart ? `
                        <button class="restart-btn" onclick="restartFromTask('${t.task_name}')" title="Restart from here">
                            <i class="fas fa-redo-alt"></i>
                        </button>
                    ` : ''}
                </div>
            </div>
        `;
    });
    container.innerHTML = html;
}

function updateTaskProgress(data) {
    const node = document.getElementById(`task-node-${data.task_name}`);
    if (node) {
        node.className = `task-node running`;
        const icon = node.querySelector('.node-icon');
        if (icon) icon.className = 'node-icon fas fa-spinner fa-spin';
    }
}

async function restartFromTask(taskName) {
    if (!confirm(`Restart from '${taskName}'? This will re-run this step and all following steps.`)) return;
    
    // Clear the terminal logs when restarting
    const term = document.getElementById('processing-term');
    if (term) term.innerHTML = '';
    
    // Reset the task chain UI visually
    const container = document.getElementById('granular-task-list');
    if (container) {
        container.innerHTML = '<div class="empty-state">Restarting workflow...</div>';
    }
    
    // Reset progress
    const progressBar = document.getElementById('main-progress-bar');
    if (progressBar) progressBar.style.width = '0%';
    
    const progressText = document.getElementById('processing-percent');
    if (progressText) progressText.innerText = '0%';
    
    const statusText = document.getElementById('processing-status-text');
    if (statusText) statusText.innerText = 'Restarting...';
    
    try {
        await postJSON(`/api/projects/${currentTaskId}/restart`, { from_task: taskName });
        
        // Reload to get fresh state
        setTimeout(() => {
            if (currentTaskId) loadAndMonitor(currentTaskId);
        }, 500);
        
    } catch (e) {
        alert("Restart failed: " + e.message);
        logToTerm(`Restart failed: ${e.message}`, 'error');
    }
}

function updateMainProgress(data) {
    const bar = document.getElementById('main-progress-bar');
    if(bar) bar.style.width = `${data.percent}%`;
    
    const txt = document.getElementById('processing-percent');
    if(txt) txt.innerText = `${data.percent}%`;
    
    const status = document.getElementById('processing-status-text');
    if(status) status.innerText = data.message || '';
}

function logToTerm(msg, style='info') {
    const term = document.getElementById('processing-term');
    if (!term) return;
    
    const div = document.createElement('div');
    div.className = `log-line ${style}`;
    
    // Add timestamp
    const time = new Date().toLocaleTimeString();
    div.innerText = `[${time}] ${msg}`;
    
    term.appendChild(div);
    term.scrollTop = term.scrollHeight;
    
    // Limit log lines to prevent memory issues
    while (term.children.length > 500) {
        term.removeChild(term.firstChild);
    }
}

// --- MERGE UI FUNCTIONS ---

function resetMergeUI() {
    speakerMergeState = { selected: new Set(), groups: [] };
    const checkboxes = document.getElementById('merge-checkboxes');
    const preview = document.getElementById('merge-preview');
    const groupsDisplay = document.getElementById('merge-groups-display');
    const status = document.getElementById('merge-status');
    const applyBtn = document.getElementById('apply-merge-btn');
    
    if (checkboxes) checkboxes.innerHTML = '';
    if (preview) preview.style.display = 'none';
    if (groupsDisplay) groupsDisplay.innerHTML = '';
    if (status) status.innerText = '';
    if (applyBtn) {
        applyBtn.disabled = true;
        applyBtn.innerHTML = '<i class="fas fa-code-branch"></i> Merge Selected';
    }
}

function setupMergeUI(speakerIds) {
    resetMergeUI();
    
    const checkboxes = document.getElementById('merge-checkboxes');
    const status = document.getElementById('merge-status');
    if (!checkboxes) return;
    
    // Create checkboxes for each speaker
    speakerIds.forEach(id => {
        const label = document.createElement('label');
        label.className = 'merge-checkbox';
        label.dataset.id = id;
        
        label.innerHTML = `
            <input type="checkbox" value="${id}">
            <span>Speaker ${parseInt(id) + 1}</span>
        `;
        
        label.addEventListener('click', (e) => {
            e.preventDefault();
            const cb = label.querySelector('input');
            cb.checked = !cb.checked;
            label.classList.toggle('checked', cb.checked);
            toggleSpeakerSelection(id);
        });
        
        checkboxes.appendChild(label);
    });
    
    if (status) {
        status.innerText = `${speakerIds.length} speaker${speakerIds.length !== 1 ? 's' : ''} detected`;
    }
}

function toggleSpeakerSelection(speakerId) {
    if (speakerMergeState.selected.has(speakerId)) {
        speakerMergeState.selected.delete(speakerId);
    } else {
        speakerMergeState.selected.add(speakerId);
    }
    
    // Enable/disable merge button
    const applyBtn = document.getElementById('apply-merge-btn');
    if (applyBtn) {
        applyBtn.disabled = speakerMergeState.selected.size < 2;
    }
    
    updateMergeStatus();
}

function updateMergeStatus() {
    const status = document.getElementById('merge-status');
    if (!status) return;
    
    const count = speakerMergeState.selected.size;
    if (count === 0) {
        status.innerText = 'Select speakers to merge';
    } else if (count === 1) {
        status.innerText = 'Select at least one more speaker';
    } else {
        status.innerText = `${count} speakers selected for merge`;
    }
}

function applySpeakerMerge() {
    const selected = Array.from(speakerMergeState.selected);
    if (selected.length < 2) return;
    
    // Add as a new merge group
    speakerMergeState.groups.push([...selected]);
    
    // Clear selection
    speakerMergeState.selected.clear();
    
    // Update UI
    document.querySelectorAll('.merge-checkbox').forEach(cb => {
        cb.classList.remove('checked');
        cb.querySelector('input').checked = false;
    });
    
    const applyBtn = document.getElementById('apply-merge-btn');
    if (applyBtn) applyBtn.disabled = true;
    
    updateMergeStatus();
    renderMergeGroups();
    updateSpeakerCardsForMerge();
}

function renderMergeGroups() {
    const preview = document.getElementById('merge-preview');
    const groupsDisplay = document.getElementById('merge-groups-display');
    if (!preview || !groupsDisplay) return;
    
    if (speakerMergeState.groups.length === 0) {
        preview.style.display = 'none';
        return;
    }
    
    preview.style.display = 'block';
    groupsDisplay.innerHTML = '';
    
    speakerMergeState.groups.forEach((group, idx) => {
        const tag = document.createElement('div');
        tag.className = 'merge-group-tag';
        
        const speakerNames = group.map(id => `Spk ${parseInt(id) + 1}`).join(' + ');
        tag.innerHTML = `
            <i class="fas fa-object-group"></i>
            <span>${speakerNames}</span>
            <span class="remove-merge" onclick="removeMergeGroup(${idx})" title="Unmerge">
                <i class="fas fa-times"></i>
            </span>
        `;
        
        groupsDisplay.appendChild(tag);
    });
}

window.removeMergeGroup = function(groupIdx) {
    speakerMergeState.groups.splice(groupIdx, 1);
    renderMergeGroups();
    updateSpeakerCardsForMerge();
};

function updateSpeakerCardsForMerge() {
    // Reset all cards
    document.querySelectorAll('.speaker-card').forEach(card => {
        card.classList.remove('merged-master', 'merged-into');
        card.style.position = '';
    });
    
    // Apply merge styling
    speakerMergeState.groups.forEach(group => {
        // First speaker in group is the "master" (keeps the voice)
        const masterId = group[0];
        const masterCard = document.querySelector(`.speaker-card[data-id="${masterId}"]`);
        if (masterCard) masterCard.classList.add('merged-master');
        
        // Others are "merged into" the master
        group.slice(1).forEach(id => {
            const card = document.querySelector(`.speaker-card[data-id="${id}"]`);
            if (card) {
                card.classList.add('merged-into');
                card.style.position = 'relative';
            }
        });
    });
}

function renderSpeakerConfig(config) {
    const grid = document.getElementById('speaker-grid');
    if(!grid) return;
    
    const entries = Object.entries(config);
    if (entries.length === 0) {
        grid.innerHTML = '<div class="empty-state">No speakers detected</div>';
        return;
    }
    
    grid.innerHTML = entries.map(([id, info]) => {
        // Check if this speaker is merged into another
        const mergedGroup = speakerMergeState.groups.find(g => g.includes(id));
        const isMaster = mergedGroup && mergedGroup[0] === id;
        const isMerged = mergedGroup && !isMaster;
        
        return `
        <div class="speaker-card ${isMaster ? 'merged-master' : ''} ${isMerged ? 'merged-into' : ''}" 
             data-id="${id}" style="${isMerged ? 'position: relative;' : ''}">
            <div class="spk-header">
                <span class="avatar">${parseInt(id) + 1}</span>
                <label class="checkbox-label">
                    <input type="checkbox" class="spk-toggle" ${isMerged ? 'disabled' : 'checked'}>
                    <span class="checkmark"></span>
                    <span class="checkbox-text">
                        <span><i class="fas fa-volume-up"></i> Dub This Speaker</span>
                        <span class="help-text">${isMerged ? 'Merged with Speaker ' + (parseInt(mergedGroup[0]) + 1) : 'Voice will be cloned'}</span>
                    </span>
                </label>
            </div>
            <div class="spk-controls">
                <input type="text" class="spk-name" value="${info.name || 'Speaker '+ (parseInt(id)+1)}" placeholder="Speaker Name" ${isMerged ? 'disabled' : ''}>
            </div>
            <audio controls src="${info.sample_path}" style="width:100%; margin-top:10px;"></audio>
        </div>
    `}).join('');
    
    // Re-apply merge styling
    updateSpeakerCardsForMerge();
}

async function startTranslationPhase() {
    const speakers = {};
    const cards = document.querySelectorAll('.speaker-card');
    
    if (cards.length === 0) {
        alert('No speakers configured');
        return;
    }
    
    // Build merge mapping and speaker config
    const mergeMap = {};  // merged speaker -> master speaker
    speakerMergeState.groups.forEach(group => {
        const master = group[0];
        // Map all non-master speakers in group to the master
        group.slice(1).forEach(id => {
            mergeMap[id] = master;
        });
    });
    
    cards.forEach(card => {
        const id = card.dataset.id;
        const name = card.querySelector('.spk-name').value;
        const isDubbing = card.querySelector('.spk-toggle').checked;
        
        // Check if this speaker is merged into another
        const mergedInto = mergeMap[id];
        
        if (mergedInto) {
            // Merged speaker: mark as remove (segments will use master's voice)
            // but record the merge info for the backend
            speakers[id] = { 
                name: name || `Speaker ${parseInt(id) + 1}`, 
                action: 'remove',
                merged_into: mergedInto  // Tell backend which master to use
            };
        } else {
            // Master speaker or unmerged speaker
            // Check if this speaker has others merged into it
            const mergedSpeakers = [];
            speakerMergeState.groups.forEach(group => {
                if (group[0] === id) {
                    mergedSpeakers.push(...group.slice(1));
                }
            });
            
            speakers[id] = { 
                name: name || `Speaker ${parseInt(id) + 1}`, 
                action: isDubbing ? 'dub' : 'remove',
                merged_speakers: mergedSpeakers  // Tell backend which speakers are merged into this one
            };
        }
    });

    const btn = document.getElementById('start-dub-btn');
    const originalContent = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Starting...';
    
    try {
        await postJSON(`/api/projects/${currentTaskId}/validate`, { speakers });
        // WebSocket will handle the transition to processing state
        showWizardStep('state-processing');
    } catch(err) {
        alert("Error: " + err.message);
        logToTerm(`Validation error: ${err.message}`, 'error');
        btn.disabled = false;
        btn.innerHTML = originalContent;
    }
}
