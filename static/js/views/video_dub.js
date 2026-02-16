import { postFormData, getJSON, postJSON } from '../api.js';
import { doSwitchView } from '../app.js';

window.openMonitor = openMonitor;
window.restartFromTask = restartFromTask;
window.startNewProject = startNewProject;

let currentWs = null;
let currentTaskId = null;
let downloadedVideoPath = null;
let downloadedVideoFilename = null;  // Store filename for download

// Speaker merge state
let speakerMergeState = {
    selected: new Set(),
    groups: [] // Array of arrays: [[0, 2], [1]] means 0&2 merged, 1 separate
};

// Settings persistence key
const SETTINGS_KEY = 'voicedub_project_settings';

// Default settings
const DEFAULT_SETTINGS = {
    src_lang: 'auto',
    tgt_lang: 'en',
    tts_engine: 'f5',
    separate_audio: false
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

    // 4. Bind Validation Confirm - will be dynamically rebound based on phase
    const startDubBtn = document.getElementById('start-dub-btn');
    if (startDubBtn) {
        // Initial binding for transcription review
        startDubBtn.addEventListener('click', startTranslationPhase);
    }
    
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

    // 7. Bind settings change listeners for persistence
    bindSettingsPersistence();
}

// NEW: Bind settings change listeners to save to localStorage
function bindSettingsPersistence() {
    const srcLang = document.getElementById('src-lang');
    const tgtLang = document.getElementById('tgt-lang');
    const ttsEngine = document.getElementById('tts-engine');
    const separateAudio = document.getElementById('separate-audio');

    const saveSettings = () => {
        const settings = {
            src_lang: srcLang ? srcLang.value : 'auto',
            tgt_lang: tgtLang ? tgtLang.value : 'en',
            tts_engine: ttsEngine ? ttsEngine.value : 'f5',
            separate_audio: separateAudio ? separateAudio.checked : false
        };
        
        try {
            localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
            console.log('Settings saved:', settings);
        } catch (e) {
            console.warn('Could not save settings:', e);
        }
    };

    // Add change listeners
    if (srcLang) srcLang.addEventListener('change', saveSettings);
    if (tgtLang) tgtLang.addEventListener('change', saveSettings);
    if (ttsEngine) ttsEngine.addEventListener('change', saveSettings);
    if (separateAudio) separateAudio.addEventListener('change', saveSettings);
}

// NEW: Load saved settings from localStorage
function loadSavedSettings() {
    try {
        const saved = localStorage.getItem(SETTINGS_KEY);
        if (!saved) {
            console.log('No saved settings found, using defaults');
            return DEFAULT_SETTINGS;
        }
        
        const settings = JSON.parse(saved);
        console.log('Settings loaded:', settings);
        return { ...DEFAULT_SETTINGS, ...settings };
    } catch (e) {
        console.warn('Could not load settings:', e);
        return DEFAULT_SETTINGS;
    }
}

// NEW: Apply loaded settings to form fields
function applySettingsToForm(settings) {
    const srcLang = document.getElementById('src-lang');
    const tgtLang = document.getElementById('tgt-lang');
    const ttsEngine = document.getElementById('tts-engine');
    const separateAudio = document.getElementById('separate-audio');

    if (srcLang && settings.src_lang) srcLang.value = settings.src_lang;
    if (tgtLang && settings.tgt_lang) tgtLang.value = settings.tgt_lang;
    if (ttsEngine && settings.tts_engine) ttsEngine.value = settings.tts_engine;
    if (separateAudio && settings.separate_audio !== undefined) {
        separateAudio.checked = settings.separate_audio;
    }
}

export function onShow() {
    console.log('VideoDub view shown. Current task ID:', currentTaskId);
    
    // If we have an active task ID, we're in monitor mode - resume monitoring
    if (currentTaskId) {
        loadAndMonitor(currentTaskId);
    } else {
        // NEW: Load and apply saved settings when showing the form
        const settings = loadSavedSettings();
        applySettingsToForm(settings);
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
        youtubeStatus.innerHTML = '';
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
    
    // NEW: Load and apply saved settings
    const settings = loadSavedSettings();
    applySettingsToForm(settings);
    
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

async function downloadYouTubeVideo() {
    if (!downloadedVideoPath) {
        alert('No video available to download');
        return;
    }
    
    try {
        // Use the backend download endpoint
        const downloadUrl = `/api/youtube/download-file?file_path=${encodeURIComponent(downloadedVideoPath)}`;
        
        // Create temporary link and trigger download
        const link = document.createElement('a');
        link.href = downloadUrl;
        link.download = downloadedVideoFilename || 'youtube_video.mp4';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
    } catch (err) {
        console.error('Download failed:', err);
        alert('Download failed: ' + err.message);
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
            downloadedVideoFilename = res.filename || res.title || 'youtube_video.mp4';
            statusEl.innerHTML = `
                <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
                    <span>✅ Downloaded: ${res.title}</span>
                    <button type="button" id="download-youtube-video-btn" class="btn-secondary btn-sm">
                        <i class="fas fa-download"></i> Download Video
                    </button>
                </div>
            `;
            statusEl.style.color = 'var(--success)';
            
            // Bind download button
            const downloadBtn = document.getElementById('download-youtube-video-btn');
            if (downloadBtn) {
                downloadBtn.addEventListener('click', downloadYouTubeVideo);
            }
            
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
        downloadedVideoFilename = null;
        
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
    // Don't close existing connection if it's the same task and still open
    if (currentWs) {
        if (currentWs.readyState === WebSocket.OPEN || currentWs.readyState === WebSocket.CONNECTING) {
            console.log('WebSocket already connected or connecting, skipping new connection');
            return;
        }
        // Only close if it's a different task or already closing/closed
        try {
            currentWs.close();
        } catch (e) {
            console.log('Error closing old WebSocket:', e);
        }
        currentWs = null;
    }
    
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${proto}//${window.location.host}/ws/${taskId}`;
    console.log('Connecting WebSocket to:', wsUrl);
    
    currentWs = new WebSocket(wsUrl);
    let reconnectScheduled = false;
    
    currentWs.onopen = () => {
        console.log('WebSocket connected successfully');
        reconnectScheduled = false;
    };
    
    currentWs.onmessage = (e) => {
        try {
            const msg = JSON.parse(e.data);
            console.log('WS message:', msg.type, msg);
            
            if (msg.type === 'state_sync') {
                // Initial state received
                renderTaskState(msg.data);
            } else if (msg.type === 'task_progress') {
                updateTaskProgress(msg.data);
            } else if (msg.type === 'status_update') {
                renderTaskState(msg.data);
            } else if (msg.type === 'progress') {
                updateMainProgress(msg.data);
            } else if (msg.type === 'log') {
                logToTerm(msg.data.message, msg.data.style);
            } else if (msg.type === 'pong') {
                // Keepalive response, no action needed
            } else if (msg.type === 'transcription_ready') {
                showTranscriptionReview(msg.data);
            } else if (msg.type === 'translation_ready') {
                showTranslationReview(msg.data);
            } else if (msg.type === 'transcription_validated') {
                logToTerm('Transcription validated, starting translation...', 'success');
                // The server is now running translation, we should wait for translation_ready
                // Don't transition UI yet - wait for translation results
            } else if (msg.type === 'translation_validated') {
                // Translation was validated by user, now starting TTS
                logToTerm('Translation validated, starting voice synthesis...', 'success');
                showWizardStep('state-processing');
            } else if (msg.type === 'validation_accepted') {
                showWizardStep('state-processing');
            }
        } catch (err) {
            console.error('Error processing WS message:', err);
        }
    };
    
    currentWs.onerror = (err) => {
        console.error('WebSocket error:', err);
        // Don't show error to user immediately, let onclose handle reconnection
    };
    
    currentWs.onclose = (event) => {
        console.log('WebSocket closed', event.code, event.reason);
        
        // Don't reconnect if:
        // 1. No current task
        // 2. Task is completed
        // 3. Already scheduled a reconnect
        // 4. Normal closure (code 1000 or 1001)
        const isCompleted = document.getElementById('state-result')?.classList.contains('active');
        const isNormalClose = event.code === 1000 || event.code === 1001;
        
        if (!currentTaskId || isCompleted || reconnectScheduled || isNormalClose) {
            console.log('Not reconnecting WebSocket:', {currentTaskId, isCompleted, reconnectScheduled, isNormalClose});
            return;
        }
        
        reconnectScheduled = true;
        logToTerm('Connection lost - reconnecting in 3s...', 'warning');
        
        setTimeout(() => {
            reconnectScheduled = false;
            if (currentTaskId && currentWs?.readyState !== WebSocket.OPEN) {
                console.log('Attempting WebSocket reconnect...');
                connectWebSocket(currentTaskId);
            }
        }, 3000);
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

// NEW: Show transcription review UI
// NEW: Show transcription review UI
function showTranscriptionReview(data) {
    showWizardStep('state-validation');
    
    // Update header to indicate transcription review
    const header = document.querySelector('#state-validation h3');
    if (header) {
        header.innerHTML = '<i class="fas fa-file-audio"></i> Review Transcriptions';
    }
    
    // Render editable transcription table
    const grid = document.getElementById('speaker-grid');
    if (!grid) return;
    
    const segments = data.segments || [];
    const speakerConfig = data.speaker_config || {};
    
    // Group segments by speaker
    const bySpeaker = {};
    segments.forEach(seg => {
        const sid = seg.speaker_id;
        if (!bySpeaker[sid]) bySpeaker[sid] = [];
        bySpeaker[sid].push(seg);
    });
    
    // Build review UI
    let html = '<div class="transcription-review">';
    html += '<p class="subtitle" style="margin-bottom:20px;">Review and edit what each speaker said. You can correct transcription errors before translation.</p>';
    
    Object.entries(bySpeaker).forEach(([sid, segs]) => {
        const spkInfo = speakerConfig[sid] || { name: `Speaker ${parseInt(sid)+1}` };
        
        html += `
            <div class="speaker-transcription-group" data-speaker="${sid}">
                <div class="spk-header" style="margin-bottom:15px;">
                    <span class="avatar">${parseInt(sid) + 1}</span>
                    <input type="text" class="spk-name" value="${spkInfo.name}" 
                           style="flex:1; background:var(--bg-panel); border:1px solid var(--border); color:var(--text-main); padding:8px; border-radius:4px;">
                    <audio controls src="${spkInfo.sample_path}" style="width:200px; margin-left:10px;"></audio>
                </div>
                <div class="segments-list">
        `;
        
        segs.forEach((seg, idx) => {
            const timeStr = formatTime(seg.start) + ' - ' + formatTime(seg.end);
            html += `
                <div class="segment-edit-row" data-idx="${seg.idx}" style="margin-bottom:12px; padding:12px; background:var(--bg-main); border-radius:6px; border:1px solid var(--border);">
                    <div style="display:flex; justify-content:space-between; margin-bottom:8px; font-size:0.85rem; color:var(--text-muted);">
                        <span><i class="fas fa-clock"></i> ${timeStr}</span>
                        <span>Segment ${seg.idx + 1}</span>
                    </div>
                    <label style="display:block; margin-bottom:4px; font-size:0.8rem; color:var(--text-muted);">Original Text:</label>
                    <textarea class="seg-original" data-idx="${seg.idx}" rows="2" 
                        style="width:100%; background:var(--bg-panel); border:1px solid var(--border); color:var(--text-main); padding:8px; border-radius:4px; margin-bottom:8px; font-family:inherit; resize:vertical;">${seg.original_text || ''}</textarea>
                    <div style="display:flex; gap:8px; align-items:center;">
                        <label style="font-size:0.8rem; color:var(--text-muted);">
                            <input type="checkbox" class="seg-include" checked> Include in dubbing
                        </label>
                    </div>
                </div>
            `;
        });
        
        html += '</div></div>';
    });
    
    html += '</div>';
    grid.innerHTML = html;
    
    // Update button - CRITICAL FIX: Use event delegation instead of cloning
    const startBtn = document.getElementById('start-dub-btn');
    if (startBtn) {
        // Clear existing handlers by replacing with fresh element
        const parent = startBtn.parentNode;
        const freshBtn = document.createElement('button');
        freshBtn.id = 'start-dub-btn';
        freshBtn.className = 'btn-success';
        freshBtn.innerHTML = '<i class="fas fa-language"></i> Translate & Continue';
        freshBtn.disabled = false;
        
        parent.replaceChild(freshBtn, startBtn);
        
        // Bind handler
        freshBtn.addEventListener('click', submitTranscriptionReview);
    }
}

// NEW: Show translation review UI
function showTranslationReview(data) {
    showWizardStep('state-validation');
    
    // Update header
    const header = document.querySelector('#state-validation h3');
    if (header) {
        header.innerHTML = '<i class="fas fa-language"></i> Review Translations';
    }
    
    const grid = document.getElementById('speaker-grid');
    if (!grid) return;
    
    const segments = data.segments || [];
    const tgtLang = data.target_language || 'en';
    const srcLang = data.source_language || 'auto';
    
    // Group by speaker
    const bySpeaker = {};
    segments.forEach(seg => {
        const sid = seg.speaker_id;
        if (!bySpeaker[sid]) bySpeaker[sid] = [];
        bySpeaker[sid].push(seg);
    });
    
    let html = '<div class="translation-review">';
    html += `<p class="subtitle" style="margin-bottom:20px;">Review and edit translations side-by-side. ${srcLang !== 'auto' ? `Translating from ${srcLang} to ${tgtLang}` : `Auto-detect to ${tgtLang}`}</p>`;
    
    Object.entries(bySpeaker).forEach(([sid, segs]) => {
        html += `
            <div class="speaker-translation-group" data-speaker="${sid}" style="margin-bottom:20px;">
                <div style="font-weight:600; margin-bottom:10px; color:var(--accent);">
                    <i class="fas fa-user"></i> Speaker ${parseInt(sid) + 1}
                </div>
        `;
        
        segs.forEach(seg => {
            const timeStr = formatTime(seg.start) + ' - ' + formatTime(seg.end);
            html += `
                <div class="translation-edit-row" data-idx="${seg.idx}" style="margin-bottom:12px; padding:12px; background:var(--bg-main); border-radius:6px; border:1px solid var(--border);">
                    <div style="display:flex; justify-content:space-between; margin-bottom:8px; font-size:0.85rem; color:var(--text-muted);">
                        <span><i class="fas fa-clock"></i> ${timeStr}</span>
                    </div>
                    <div class="side-by-side-container" style="display:grid; grid-template-columns:1fr 1fr; gap:12px;">
                        <div class="original-column">
                            <label style="display:block; font-size:0.8rem; color:var(--text-muted); margin-bottom:4px;">
                                <i class="fas fa-file-alt"></i> Original:
                            </label>
                            <div class="original-text" style="padding:10px; background:var(--bg-panel); border-radius:4px; color:var(--text-main); font-style:italic; min-height:60px; line-height:1.5;">
                                ${seg.original_text || '[No text]'}
                            </div>
                        </div>
                        <div class="translation-column">
                            <label style="display:block; font-size:0.8rem; color:var(--accent); margin-bottom:4px;">
                                <i class="fas fa-language"></i> Translation (${tgtLang}):
                            </label>
                            <textarea class="seg-translated" data-idx="${seg.idx}" rows="3" 
                                style="width:100%; background:var(--bg-panel); border:1px solid var(--border); color:var(--text-main); padding:10px; border-radius:4px; font-family:inherit; resize:vertical; min-height:60px; line-height:1.5;">${seg.translated_text || ''}</textarea>
                        </div>
                    </div>
                </div>
            `;
        });
        
        html += '</div>';
    });
    
    html += '</div>';
    grid.innerHTML = html;
    
    // Update button - CRITICAL FIX: Use event delegation instead of cloning
    const startBtn = document.getElementById('start-dub-btn');
    if (startBtn) {
        // Clear existing handlers by replacing with fresh element
        const parent = startBtn.parentNode;
        const freshBtn = document.createElement('button');
        freshBtn.id = 'start-dub-btn';
        freshBtn.className = 'btn-success';
        freshBtn.innerHTML = '<i class="fas fa-microphone"></i> Generate Voices';
        freshBtn.disabled = false;
        
        parent.replaceChild(freshBtn, startBtn);
        
        // Bind handler
        freshBtn.addEventListener('click', submitTranslationReview);
    }
}

// NEW: Submit transcription review
// NEW: Submit transcription review
async function submitTranscriptionReview() {
    console.log('submitTranscriptionReview called');
    
    // Collect edited transcriptions
    const editedSegments = [];
    
    document.querySelectorAll('.segment-edit-row').forEach(row => {
        const idx = parseInt(row.dataset.idx);
        const originalText = row.querySelector('.seg-original').value;
        const include = row.querySelector('.seg-include').checked;
        
        // Find speaker ID from parent group
        const speakerGroup = row.closest('.speaker-transcription-group');
        const speakerId = speakerGroup ? parseInt(speakerGroup.dataset.speaker) : 0;
        
        editedSegments.push({
            idx: idx,
            speaker_id: speakerId,
            original_text: originalText,
            include: include,
            // Preserve timing info - will be filled from original
        });
    });
    
    // Also collect speaker config updates
    const speakers = {};
    document.querySelectorAll('.speaker-transcription-group').forEach(group => {
        const sid = group.dataset.speaker;
        const nameInput = group.querySelector('.spk-name');
        speakers[sid] = {
            name: nameInput ? nameInput.value : `Speaker ${parseInt(sid)+1}`,
            action: 'dub'  // Default, can be changed
        };
    });
    
    console.log(`Collected ${editedSegments.length} edited segments, ${Object.keys(speakers).length} speakers`);
    
    // Get the button - it may have been replaced, so find it fresh
    const btn = document.getElementById('start-dub-btn');
    if (!btn) {
        console.error('Translate & Continue button not found!');
        alert('Error: Button not found');
        return;
    }
    
    const originalContent = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Starting Translation...';
    
    try {
        console.log(`Sending validate request for task ${currentTaskId}`);
        const response = await postJSON(`/api/projects/${currentTaskId}/validate`, { 
            speakers: speakers,
            edited_segments: editedSegments,
            proceed_to_translation: true
        });
        console.log('validate response:', response);
        
        // Show processing state while waiting for translation to complete
        showWizardStep('state-processing');
        logToTerm('Translation in progress...', 'info');
        
        // WebSocket will send translation_ready when complete
    } catch(err) {
        console.error('Transcription validation error:', err);
        alert("Error: " + err.message);
        logToTerm(`Validation error: ${err.message}`, 'error');
        btn.disabled = false;
        btn.innerHTML = originalContent;
    }
}

// NEW: Submit translation review
async function submitTranslationReview() {
    console.log('submitTranslationReview called');
    
    // Collect edited translations
    const editedSegments = [];
    
    document.querySelectorAll('.translation-edit-row').forEach(row => {
        const idx = parseInt(row.dataset.idx);
        const translatedText = row.querySelector('.seg-translated').value;
        const originalText = row.querySelector('.original-text').textContent.trim();
        
        editedSegments.push({
            idx: idx,
            original_text: originalText,
            translated_text: translatedText
        });
    });
    
    console.log(`Collected ${editedSegments.length} edited segments`);
    
    // Get the button - it may have been replaced, so find it fresh
    const btn = document.getElementById('start-dub-btn');
    if (!btn) {
        console.error('Generate Voices button not found!');
        alert('Error: Button not found');
        return;
    }
    
    const originalContent = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Starting Voice Synthesis...';
    
    try {
        console.log(`Sending validate-translation request for task ${currentTaskId}`);
        // Send flat structure - the backend will handle it
        const response = await postJSON(`/api/projects/${currentTaskId}/validate-translation`, { 
            edited_segments: editedSegments,
            proceed_to_tts: true
        });
        console.log('validate-translation response:', response);
        
        // WebSocket will handle transition to processing
        showWizardStep('state-processing');
        logToTerm('Voice synthesis starting...', 'info');
    } catch(err) {
        console.error('Translation validation error:', err);
        alert("Error: " + err.message);
        logToTerm(`Translation validation error: ${err.message}`, 'error');
        btn.disabled = false;
        btn.innerHTML = originalContent;
    }
}

// Legacy function - kept for compatibility
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

// Helper for time formatting
function formatTime(seconds) {
    if (!seconds || isNaN(seconds)) return '00:00';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    const ms = Math.floor((seconds % 1) * 100);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}.${ms.toString().padStart(2, '0')}`;
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
