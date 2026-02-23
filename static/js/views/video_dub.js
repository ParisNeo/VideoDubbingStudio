import { postFormData, getJSON, postJSON, formatTime } from '../api.js';
import { doSwitchView } from '../app.js';

window.openMonitor = openMonitor;
window.restartFromTask = restartFromTask;
window.startNewProject = startNewProject;

let currentWs = null;
let currentTaskId = null;
let downloadedVideoPath = null;
let downloadedVideoFilename = null;  // Store filename for download
let currentTaskState = {};           // Single source of truth for UI state

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

    // 4. Validation Confirm buttons are now dynamically created in the render functions based on the current phase
    
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
    currentTaskState = {};
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

    // Toggle shared timeline header
    const timeline = document.getElementById('task-timeline-header');
    if (timeline) {
        if (stepId === 'state-upload') {
            timeline.classList.add('hidden');
        } else {
            timeline.classList.remove('hidden');
        }
    }
}

async function downloadYouTubeVideo() {
    if (!downloadedVideoPath) {
        await window.uiAlert('No video available to download', 'Error');
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
        await window.uiAlert('Download failed: ' + err.message, 'Error');
    }
}

async function handleYouTubeDownload() {
    const urlInput = document.getElementById('youtube-url');
    const statusEl = document.getElementById('youtube-status');
    const btn = document.getElementById('download-youtube-btn');
    
    const url = urlInput.value.trim();
    if (!url) return await window.uiAlert("Please enter a YouTube URL", "Input Required");

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
        await window.uiAlert('Error starting project: ' + err.message, 'Upload Failed');
        btn.disabled = false;
        btn.innerHTML = originalBtnContent;
    }
}

async function loadAndMonitor(taskId, retryCount = 0) {
    currentTaskId = taskId;
    
    try {
        const task = await getJSON(`/api/projects/${taskId}`);
        // Populate global state immediately
        Object.assign(currentTaskState, task);
        renderTaskState(task);
        connectWebSocket(taskId);
    } catch(err) {
        // Handle transient race conditions where task isn't yet committed/visible in DB
        // Retry up to 5 times with increasing delay
        if (err.message.includes('404') && retryCount < 5) {
            const delay = 400 * (retryCount + 1);
            console.warn(`Task ${taskId} not found yet (syncing DB...), retrying in ${delay}ms...`);
            setTimeout(() => loadAndMonitor(taskId, retryCount + 1), delay);
            return;
        }
        
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
                // Merge progress into state and trigger a full re-render
                currentTaskState.phase = msg.data.phase;
                currentTaskState.progress = msg.data.percent;
                currentTaskState.message = msg.data.message;
                if (msg.data.phase === 'complete') currentTaskState.status = 'completed';
                renderTaskState(currentTaskState);
            } else if (msg.type === 'log') {
                logToTerm(msg.data.message, msg.data.style);
            } else if (msg.type === 'pong') {
                // Keepalive response, no action needed
            } else if (msg.type === 'speaker_validation_ready') {
                Object.assign(currentTaskState, msg.data);
                showSpeakerValidation(currentTaskState);
            } else if (msg.type === 'transcription_ready') {
                showTranscriptionReview(msg.data);
            } else if (msg.type === 'translation_ready') {
                showTranslationReview(msg.data);
            } else if (msg.type === 'audio_ready') {
                Object.assign(currentTaskState, msg.data);
                showAudioValidation(currentTaskState);
            } else if (msg.type === 'transcription_validated') {
                Object.assign(currentTaskState, msg.data);
                logToTerm('Transcription validated, starting translation...', 'success');
                renderTaskState(currentTaskState);
            } else if (msg.type === 'translation_validated') {
                Object.assign(currentTaskState, msg.data);
                logToTerm('Translation validated, starting voice synthesis...', 'success');
                renderTaskState(currentTaskState);
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

async function renderTaskState(task) {
    console.log('Rendering task state:', task.status, task.phase);
    Object.assign(currentTaskState, task);

    // CRITICAL: If we are in a validation phase but segments are missing, 
    // fetch the full task object from the API to ensure we have all data.
    const isValidationPhase = task.status === 'awaiting_validation' || task.phase?.includes('awaiting');
    const needsData = isValidationPhase && !task.segments && !task.transcribed_segments && !task.speaker_config;
    
    if (needsData && task.task_id) {
        console.log("Validation phase detected with missing data, fetching full task state...");
        try {
            const fullTask = await getJSON(`/api/projects/${task.task_id}`);
            Object.assign(currentTaskState, fullTask);
        } catch (e) { 
            // Silent error here because WebSocket messages come frequently and will re-trigger this
            console.debug("Transient error fetching task details during state sync:", e); 
        }
    }

    // Sync progress bar
    const progressBar = document.getElementById('main-progress-bar');
    const progressText = document.getElementById('processing-percent');
    const statusText = document.getElementById('processing-status-text');

    let pct = currentTaskState.progress || 0;
    if (currentTaskState.status === 'completed' || currentTaskState.phase === 'complete') pct = 100;

    if (progressBar) progressBar.style.width = `${pct}%`;
    if (progressText) progressText.innerText = `${pct}%`;
    if (statusText && currentTaskState.message) statusText.innerText = currentTaskState.message;

    // Update workflow chain (Always use static chain as it represents the true live phases)
    renderStaticChain(currentTaskState);

    // CRITICAL: Ensure timeline is visible if we are processing (not upload)
    const timeline = document.getElementById('task-timeline-header');
    if (timeline && currentTaskState.phase !== 'init' && currentTaskState.status !== 'pending') {
        timeline.classList.remove('hidden');
    }

    // Handle Granular Validation Phases
    const phase = currentTaskState.phase;
    const status = currentTaskState.status;
    const hasSegments = (currentTaskState.segments && currentTaskState.segments.length > 0) || 
                        (currentTaskState.transcribed_segments && currentTaskState.transcribed_segments.length > 0);
    
    console.log(`UI State Check: phase=${phase}, status=${status}, hasSegments=${hasSegments}`);

    // Handle Granular Validation Phases
    // Fix: Use jumpToPhaseReview to ensure consistent UI routing when returning to a project
    if (phase === 'awaiting_transcription_review') {
        jumpToPhaseReview('awaiting_transcription_review');
    }
    else if (phase === 'awaiting_speaker_validation' || (status === 'awaiting_validation' && !hasSegments)) {
        jumpToPhaseReview('awaiting_speaker_validation');
    } 
    else if (phase === 'awaiting_translation_review') {
        jumpToPhaseReview('awaiting_translation_review');
    }
    else if (phase === 'awaiting_audio_validation') {
        jumpToPhaseReview('awaiting_audio_validation');
    }
    else if (status === 'awaiting_validation') {
        // Generic fallback: check data presence
        if (hasSegments) jumpToPhaseReview('awaiting_transcription_review');
        else jumpToPhaseReview('awaiting_speaker_validation');
    }
    else if (currentTaskState.status === 'completed' || phase === 'complete') {
        showTranslationReview({
            segments: currentTaskState.segments || [],
            target_language: currentTaskState.tgt_lang || 'en',
            source_language: currentTaskState.src_lang || 'auto'
        });
    }
    else if (phase === 'awaiting_audio_validation') {
        showAudioValidation(currentTaskState);
    }
    if (currentTaskState.status === 'completed' || phase === 'complete') {
        showWizardStep('state-result');
        const finalVideo = document.getElementById('final-video');
        const downloadBtn = document.getElementById('download-btn');
        
        if (finalVideo && currentTaskState.output_path) {
            // Normalize path for the browser (fix Windows backslashes)
            let webPath = currentTaskState.output_path.replace(/\\/g, '/');
            // Ensure path starts with /
            if (!webPath.startsWith('/')) webPath = '/' + webPath;
            finalVideo.src = webPath;
            finalVideo.load();
        }
        
        if (downloadBtn) {
            // Force the button to use the API download endpoint instead of the raw path
            downloadBtn.href = `/api/projects/${currentTaskId}/download`;
        }
    } else if (currentTaskState.status === 'failed' || currentTaskState.status === 'error') {
        logToTerm(`FAILED: ${currentTaskState.error_message || 'Unknown error'}`, 'error');
        if (statusText) {
            statusText.innerHTML = `<span style="color:var(--danger)">FAILED: ${currentTaskState.error_message || 'Unknown error'}</span>`;
        }
    } else {
        const processingStep = document.getElementById('state-processing');
        if (processingStep && processingStep.classList.contains('hidden')) {
            showWizardStep('state-processing');
        }
    }
}

function renderStaticChain(task) {
    const container = document.getElementById('granular-task-list');
    if (!container) return;
    
    // Updated granular phases
    const phases = [
        { id: 'identifying', name: 'Speaker Diarization' },
        { id: 'awaiting_speaker_validation', name: 'Speaker Confirmation' },
        { id: 'transcribing', name: 'Transcription' },
        { id: 'awaiting_transcription_review', name: 'Transcription Review' },
        { id: 'translating', name: 'Translation' },
        { id: 'awaiting_translation_review', name: 'Translation Review' },
        { id: 'synthesizing', name: 'Voice Synthesis' },
        { id: 'awaiting_audio_validation', name: 'Audio Review' },
        { id: 'recomposing', name: 'Final Assembly' },
        { id: 'complete', name: 'Completed' }
    ];

    let ph = task.phase || 'init';
    let currentPhaseIdx = -1;
    
    // Map backend phases to UI timeline index
    const phaseMap = {
        'init': 0,
        'extraction': 0,
        'identifying': 0, 
        'samples': 0,
        'awaiting_speaker_validation': 1,
        'transcribing': 2, 'running_transcription': 2,
        'awaiting_transcription_review': 3,
        'translating': 4, 'running_translation': 4,
        'awaiting_translation_review': 5,
        'synthesizing': 6, 'tts_synthesis': 6, 'running_synthesis': 6,
        'awaiting_audio_validation': 7,
        'recomposing': 8,
        'complete': 9, 'completed': 9
    };
    
    currentPhaseIdx = phaseMap[ph] !== undefined ? phaseMap[ph] : 0;
    if (task.status === 'completed') currentPhaseIdx = 9;

    let html = '';
    phases.forEach((phaseObj, idx) => {
        let statusClass = 'pending';
        let iconClass = 'fa-circle';
        
        if (idx < currentPhaseIdx) {
            statusClass = 'completed';
            iconClass = 'fa-check';
        } else if (idx === currentPhaseIdx) {
            // Highlight this as the current logical step
            statusClass += ' active-step';
            if (task.status === 'failed' || task.status === 'error') {
                statusClass = 'failed';
                iconClass = 'fa-times';
            } else if (task.status === 'paused') {
                statusClass = 'skipped';
                iconClass = 'fa-pause';
            } else if (idx === 2 || idx === 4) { 
                // UI review states
                if (task.status === 'awaiting_validation' || task.phase === 'awaiting_translation_review') {
                    statusClass = 'skipped'; 
                    iconClass = 'fa-exclamation';
                } else {
                    statusClass = 'running';
                    iconClass = 'fa-spinner fa-spin';
                }
            } else if (task.status === 'completed' || task.phase === 'complete') {
                statusClass = 'completed';
                iconClass = 'fa-check';
            } else {
                statusClass = 'running';
                iconClass = 'fa-spinner fa-spin';
            }
        }
        
        const showRestart = (statusClass === 'completed' || statusClass === 'failed') && 
                            ['init', 'identifying', 'translating', 'tts_synthesis', 'recomposing'].includes(phaseObj.id);
        
        html += `
            <div id="task-node-${phaseObj.id}" class="task-node ${statusClass}" onclick="jumpToPhaseReview('${phaseObj.id}')">
                <div class="node-icon-wrapper">
                    <i class="node-icon fas ${iconClass}"></i>
                </div>
                <div class="node-content">
                    <div>
                        <span class="node-title">${phaseObj.name}</span>
                        ${idx === currentPhaseIdx && task.message ? `<span class="node-status-text">- ${task.message}</span>` : ''}
                    </div>
                    ${showRestart ? `
                        <button class="restart-btn" onclick="event.stopPropagation(); restartFromTask('${phaseObj.id}')" title="Restart from here">
                            <i class="fas fa-redo-alt"></i>
                        </button>
                    ` : ''}
                </div>
            </div>
        `;
    });
    container.innerHTML = html;
}

// Removed legacy renderChain and updateTaskProgress functions to prevent UI freezing

// Ensure accessible globally
window.restartFromTask = restartFromTask;
window.jumpToPhaseReview = jumpToPhaseReview;

function jumpToPhaseReview(phaseId) {
    console.log(`Jumping to review for: ${phaseId}`);
    
    // Highlight the current selection in the chain
    document.querySelectorAll('.task-node').forEach(n => n.classList.remove('active-step'));
    const node = document.getElementById(`task-node-${phaseId}`);
    if (node) node.classList.add('active-step');

    const hasTranscription = (currentTaskState.segments?.length > 0) || (currentTaskState.transcribed_segments?.length > 0);

    switch(phaseId) {
        case 'identifying':
        case 'awaiting_speaker_validation':
            if (currentTaskState.speaker_config) showSpeakerValidation(currentTaskState);
            break;
            
        case 'transcribing':
        case 'awaiting_transcription_review':
            if (hasTranscription) showTranscriptionReview(currentTaskState);
            else logToTerm("Transcription data not yet available for this phase", "warning");
            break;
            
        case 'translating':
        case 'awaiting_translation_review':
            if (hasTranscription) {
                showTranslationReview({
                    segments: currentTaskState.segments || [],
                    target_language: currentTaskState.tgt_lang || 'en',
                    source_language: currentTaskState.src_lang || 'auto'
                });
            } else logToTerm("Translation data not yet available", "warning");
            break;
            
        case 'synthesizing':
        case 'awaiting_audio_validation':
            if (currentTaskState.segments) showAudioValidation(currentTaskState);
            break;
            
        case 'recomposing':
        case 'complete':
            showWizardStep('state-result');
            break;
            
        default:
            showWizardStep('state-processing');
    }
}

async function restartFromTask(taskName) {
    // Map UI static step IDs to backend phases
    const phaseMap = {
        'init': 'init',
        'identifying': 'identifying',
        'translating': 'running_translation', // Route directly to translation runner
        'tts_synthesis': 'tts_synthesis',
        'recomposing': 'recomposing'
    };
    
    const backendPhase = phaseMap[taskName] || 'init';

    const confirmed = await window.uiConfirm(`Restart from '${taskName}'? This will re-run this step and all following steps.`, 'Restart Workflow');
    if (!confirmed) return;
    
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
        await postJSON(`/api/projects/${currentTaskId}/restart`, { from_phase: backendPhase });
        
        // Reload to get fresh state
        setTimeout(() => {
            if (currentTaskId) loadAndMonitor(currentTaskId);
        }, 500);
        
    } catch (e) {
        await window.uiAlert("Restart failed: " + e.message, "Error");
        logToTerm(`Restart failed: ${e.message}`, 'error');
    }
}

// Removed updateMainProgress as renderTaskState handles all UI syncing now

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
    const groupName = document.getElementById('merge-group-name').value.trim() || `Merged Group ${speakerMergeState.groups.length + 1}`;
    
    if (selected.length < 2) return;
    
    // Add as a new merge group with name
    speakerMergeState.groups.push({
        ids: [...selected],
        name: groupName
    });
    
    // Reset name input
    document.getElementById('merge-group-name').value = '';
    
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
        
        const idChain = group.ids.map(id => `Spk ${parseInt(id) + 1}`).join(' + ');
        tag.innerHTML = `
            <i class="fas fa-object-group"></i>
            <b>${group.name}</b> <small style="opacity:0.8; margin-left:5px">(${idChain})</small>
            <span class="remove-merge" onclick="removeMergeGroup(${idx})" title="Unmerge" style="margin-left:8px; cursor:pointer;">
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

// NEW: Show translation review UI
function showTranslationReview(data) {
    showWizardStep('state-validation');
    
    // Hide merge section and subtitle for translation review
    const mergeSection = document.querySelector('.merge-section');
    const subtitleText = document.querySelector('#state-validation > .subtitle');
    if (mergeSection) mergeSection.style.display = 'none';
    if (subtitleText) subtitleText.style.display = 'none';
    
    // Update header
    const header = document.querySelector('#state-validation h3');
    if (header) {
        header.innerHTML = '<i class="fas fa-language"></i> Review Translations';
    }
    
    const grid = document.getElementById('speaker-grid');
    if (!grid) return;
    
    // Use full-width list mode
    grid.className = 'review-list-container';
    
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
                    <audio controls src="/api/projects/${currentTaskId}/preview/${seg.idx}" style="width:100%; height:30px; margin-bottom:8px;"></audio>
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
    
    // Update header actions container with retry and generate buttons
    const headerActions = document.querySelector('#state-validation .header-actions');
    if (headerActions) {
        headerActions.innerHTML = `
            <button onclick="showWizardStep('state-processing')" class="btn-icon" title="View Logs">
                <i class="fas fa-terminal"></i>
            </button>
            <button id="retry-translation-btn" class="btn-secondary">
                <i class="fas fa-redo"></i> Retry Translation
            </button>
            <button id="start-dub-btn" class="btn-success">
                <i class="fas fa-microphone"></i> Generate Voices
            </button>
        `;
        document.getElementById('start-dub-btn').addEventListener('click', submitTranslationReview);
        document.getElementById('retry-translation-btn').addEventListener('click', () => {
            restartFromTask('translating');
        });
    }
}

// NEW: Submit transcription review
// NEW: Submit transcription review
// 1. Speaker Validation UI
async function showSpeakerValidation(data) {
    console.log("Showing speaker validation with data:", data);
    showWizardStep('state-validation');
    const header = document.querySelector('#state-validation h3');
    if (header) header.innerHTML = '<i class="fas fa-user-check"></i> Step 1: Confirm Speakers';
    
    // Show merge, hide others
    const mergeSection = document.querySelector('.merge-section');
    const subtitleText = document.querySelector('#state-validation > .subtitle');
    if (mergeSection) mergeSection.style.display = 'block';
    if (subtitleText) subtitleText.style.display = 'block';
    
    const grid = document.getElementById('speaker-grid');
    if(!grid) return;
    
    grid.className = 'speaker-grid';
    // Robustly find config in either data or data.data
    const config = data.speaker_config || data.data?.speaker_config || {};
    const entries = Object.entries(config);

    if (entries.length === 0) {
        grid.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-user-slash"></i>
                <p>No speakers detected in the audio.</p>
                <button class="btn-secondary" onclick="restartFromTask('init')">Try Again</button>
            </div>`;
        return;
    }

    grid.innerHTML = entries.map(([id, info]) => {
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
        </div>`;
    }).join('');

    setupMergeUI(Object.keys(config));
    updateSpeakerCardsForMerge();

    const actions = document.querySelector('#state-validation .header-actions');
    actions.innerHTML = `
        <button id="confirm-speakers-btn" class="btn-success">
            <i class="fas fa-check"></i> Confirm & Transcribe
        </button>
    `;
    document.getElementById('confirm-speakers-btn').onclick = submitSpeakerValidation;
}

async function submitSpeakerValidation() {
    const speakers = {};
    const mergeMap = {};
    
    speakerMergeState.groups.forEach(group => {
        const master = group[0];
        group.slice(1).forEach(id => mergeMap[id] = master);
    });
    
    document.querySelectorAll('.speaker-card').forEach(card => {
        const id = card.dataset.id;
        const name = card.querySelector('.spk-name').value;
        const isDub = card.querySelector('.spk-toggle').checked;
        const mergedInto = mergeMap[id];
        
        // Find merged children if this is a master
        const merged_speakers = [];
        speakerMergeState.groups.forEach(group => {
            if (group[0] === id) merged_speakers.push(...group.slice(1));
        });

        speakers[id] = {
            name: name,
            action: mergedInto ? 'remove' : (isDub ? 'dub' : 'remove'),
            merged_into: mergedInto,
            merged_speakers: merged_speakers
        };
    });
    
    try {
        await postJSON(`/api/projects/${currentTaskId}/validate-speakers`, { speakers });
        currentTaskState.phase = 'transcribing';
        showWizardStep('state-processing');
    } catch (e) {
        await window.uiAlert("Validation failed: " + e.message, "Error");
    }
}

// 2. Transcription Review UI
function showTranscriptionReview(data) {
    console.log("Rendering Transcription Review UI...");
    showWizardStep('state-validation');
    
    // Explicitly hide Step 1 elements
    const mergeSection = document.querySelector('.merge-section');
    const step1Subtitle = document.querySelector('#state-validation > .subtitle');
    if (mergeSection) mergeSection.style.display = 'none';
    if (step1Subtitle) step1Subtitle.style.display = 'none';
    
    const header = document.querySelector('#state-validation h3');
    if (header) header.innerHTML = '<i class="fas fa-file-alt"></i> Step 2: Review Transcription';
    
    const grid = document.getElementById('speaker-grid');
    if (!grid) return;

    // Clear old buttons immediately to prevent UI confusion
    
    const actions = document.getElementById('validation-actions-container');
    if (actions) actions.innerHTML = '';

    grid.className = 'review-list-container';
    grid.innerHTML = '<div class="empty-state"><i class="fas fa-spinner fa-spin"></i> Preparing transcript...</div>';
    
    // Look for segments in all possible locations
    let segments = data.transcribed_segments || data.segments || data.data?.transcribed_segments || data.data?.segments || [];
    let speakerConfig = data.speaker_config || data.data?.speaker_config || {};

    console.log(`Found ${Array.isArray(segments) ? segments.length : '0'} segments to render.`);

    // Handle stringified JSON from database
    if (typeof segments === 'string') {
        try { segments = JSON.parse(segments); } catch(e) { console.error("Parse segments failed", e); }
    }
    if (typeof speakerConfig === 'string') {
        try { speakerConfig = JSON.parse(speakerConfig); } catch(e) { console.error("Parse config failed", e); }
    }

    if (!Array.isArray(segments) || segments.length === 0) {
        grid.innerHTML = '<div class="empty-state">No transcription segments found.</div>';
        return;
    }
    
    // Group segments by speaker
    const bySpeaker = {};
    segments.forEach(seg => {
        const sid = seg.speaker_id;
        if (!bySpeaker[sid]) bySpeaker[sid] = [];
        bySpeaker[sid].push(seg);
    });
    
    let html = '<div class="transcription-review">';
    
    Object.entries(bySpeaker).forEach(([sid, segs]) => {
        // Find the speaker name from the config, ensuring we handle string/int ID mismatches
        let spkName = `Speaker ${parseInt(sid) + 1}`;
        if (speakerConfig && (speakerConfig[sid] || speakerConfig[parseInt(sid)])) {
            const info = speakerConfig[sid] || speakerConfig[parseInt(sid)];
            if (info.name) spkName = info.name;
        }

        html += `
            <div class="speaker-transcription-group">
                <div class="spk-header">
                    <span class="avatar">${parseInt(sid) + 1}</span>
                    <span style="font-weight:600; color: var(--accent);">${spkName}</span>
                </div>
                <div class="segments-list">
        `;
        
        segs.forEach((seg, arrayIdx) => {
            // Use seg.idx if available, otherwise fallback to index in current data array
            const lookupIdx = seg.idx !== undefined ? seg.idx : segments.indexOf(seg);
            const timeStr = formatTime(seg.start) + ' - ' + formatTime(seg.end);
            html += `
                <div class="segment-edit-row" data-idx="${lookupIdx}" data-speaker="${sid}" data-start="${seg.start}" data-end="${seg.end}">
                    <div style="display:flex; justify-content:space-between; margin-bottom:4px; font-size:0.8rem; color:var(--text-muted);">
                        <span>${timeStr}</span>
                    </div>
                    <audio controls preload="none" src="/api/projects/${currentTaskId}/preview/${lookupIdx}" style="width:100%; height:32px; margin-bottom:6px;"></audio>
                    <textarea class="seg-original" style="width:100%">${seg.original_text || ''}</textarea>
                </div>
            `;
        });
        html += '</div></div>';
    });
    html += '</div>';
    grid.innerHTML = html;
    
    // Reuse the 'actions' variable declared at the top of the function
    if (actions) {
        actions.innerHTML = `
            <button id="confirm-transcription-btn" class="btn-success">
                <i class="fas fa-language"></i> Confirm & Translate
            </button>
        `;
        const confirmBtn = document.getElementById('confirm-transcription-btn');
        if (confirmBtn) confirmBtn.onclick = submitTranscriptionReview;
    }
}

async function submitTranscriptionReview() {
    const segments = [];
    document.querySelectorAll('.segment-edit-row').forEach(row => {
        segments.push({
            idx: parseInt(row.dataset.idx),
            speaker_id: parseInt(row.dataset.speaker),
            start: parseFloat(row.dataset.start),
            end: parseFloat(row.dataset.end),
            original_text: row.querySelector('.seg-original').value,
            translated_text: "" // Clear translation if text changed
        });
    });
    
    await postJSON(`/api/projects/${currentTaskId}/validate-transcription`, { segments });
    currentTaskState.phase = 'translating';
    showWizardStep('state-processing');
}

// 3. Translation Review - Handled by specialized logic below

// 4. Audio Validation (New)
function showAudioValidation(data) {
    showWizardStep('state-validation');
    document.querySelector('.merge-section').style.display = 'none';
    
    const header = document.querySelector('#state-validation h3');
    header.innerHTML = '<i class="fas fa-volume-up"></i> Step 4: Review Audio';
    
    const grid = document.getElementById('speaker-grid');
    grid.innerHTML = `
        <div style="text-align:center; padding:20px;">
            <p>Audio segments generated. You can listen to the preview below.</p>
            <!-- Add a simple list of generated segments to play -->
            <div style="max-height:400px; overflow-y:auto; text-align:left;">
                ${(data.segments || []).map(s => `
                    <div style="padding:10px; border-bottom:1px solid var(--border);">
                        <div>Segment ${s.idx + 1} (${formatTime(s.start)}-${formatTime(s.end)})</div>
                        <audio controls src="/api/projects/${currentTaskId}/preview/${s.idx}" style="width:100%; margin-top:5px;"></audio>
                    </div>
                `).join('')}
            </div>
        </div>
    `;
    
    const actions = document.querySelector('#state-validation .header-actions');
    actions.innerHTML = `
        <button id="confirm-audio-btn" class="btn-success">
            <i class="fas fa-film"></i> Confirm & Build Video
        </button>
    `;
    document.getElementById('confirm-audio-btn').onclick = async () => {
        await postJSON(`/api/projects/${currentTaskId}/validate-audio`, {});
        currentTaskState.phase = 'recomposing';
        showWizardStep('state-processing');
    };
}

// NEW: Submit translation review
async function submitTranslationReview() {
    const edited_segments = []; // Fixed variable name to match usage below
    document.querySelectorAll('.translation-edit-row').forEach(row => {
        edited_segments.push({
            idx: parseInt(row.dataset.idx),
            original_text: row.querySelector('.original-text').textContent.trim(),
            translated_text: row.querySelector('.seg-translated').value
        });
    });
    
    const btn = document.getElementById('start-dub-btn');
    if (!btn) return;
    
    const originalContent = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Starting Voice Synthesis...';

    try {
        // Send as a wrapped object to match backend expectations
        await postJSON(`/api/projects/${currentTaskId}/validate-translation`, { 
            segments: edited_segments,
            edited_segments: edited_segments,
            proceed_to_tts: true
        });
        showWizardStep('state-processing');
    } catch(err) {
        await window.uiAlert("Error: " + err.message, 'Validation Failed');
        btn.disabled = false;
        btn.innerHTML = originalContent;
    }
}
