/* 
   VoiceDub Pro Ultimate - Frontend Logic v6.1 
   Updated to support speaker sample preview, naming, merging, and non‑autoplay audio chunks.
   Added support for sending optional speaker name mappings (JSON) with upload & YouTube requests.
*/

// -------------------------------------------------
// GLOBAL STATE
// -------------------------------------------------
const appState = {
    activeView: 'dashboard',
    activeTaskId: null,
    ws: null,
    currentScript: null,
    lastRecording: null, // Stores path of last recorded file
    mediaRecorder: null,
    recordChunks: [],
    timerInterval: null,
    // New: speaker name map for the current task
    speakerNames: {}
};

// -------------------------------------------------
// NAVIGATION
// -------------------------------------------------
function switchView(viewName) {
    // Hide all
    document.querySelectorAll('.view-section').forEach(el => el.style.display = 'none');
    
    // Show target
    const target = document.getElementById('view-' + viewName);
    if (target) target.style.display = 'block';

    // Update Sidebar
    document.querySelectorAll('.nav-btn').forEach(btn => btn.classList.remove('active'));
    
    // Try to find button matching the view
    const navBtn = document.querySelector(`.nav-btn[onclick="switchView('${viewName}')"]`);
    if (navBtn) navBtn.classList.add('active');

    // Dashboard Refresh
    if (viewName === 'dashboard') refreshDashboard();

    appState.activeView = viewName;
    
    // Persist the active view in localStorage
    try {
        localStorage.setItem('vdpu_active_view', viewName);
    } catch (e) {
        console.warn('Unable to save UI state to localStorage:', e);
    }

    // WS Cleanup if leaving monitor
    if (viewName !== 'monitor' && appState.ws) {
        appState.ws.close();
        appState.ws = null;
    }
}

function showDashboard() { switchView('dashboard'); }
function showCreate() { switchView('create'); }

// -------------------------------------------------
// SHOW PROCESS MONITOR
// -------------------------------------------------
/**
 * Open the monitor view for a given task.
 * It stores the taskId, switches to the monitor view,
 * and opens a WebSocket connection for real‑time updates.
 */
function showProcessMonitor(taskId) {
    appState.activeTaskId = taskId;
    switchView('monitor');
    connectWebSocket(taskId);
}

// -------------------------------------------------
// DASHBOARD LOGIC
// -------------------------------------------------
async function refreshDashboard() {
    const list = document.getElementById('task-list');
    if(!list) return;

    try {
        const res = await fetch('/api/tasks');
        const data = await res.json();
        const tasks = data.tasks;

        list.innerHTML = '';
        if (tasks.length === 0) {
            list.innerHTML = '<div class="empty-state">No projects found.</div>';
            return;
        }

        tasks.forEach(task => {
            const card = document.createElement('div');
            card.className = 'task-card';

            // Format Date
            let dateStr = "Unknown";
            if (task.created_at) {
                dateStr = new Date(parseFloat(task.created_at)*1000).toLocaleString();
            }
            
            // Badges
            const sepBadge = task.separate_audio ? 
                '<span class="badge badge-pink">Demucs</span>' : 
                '<span class="badge badge-blue">Smart Mix</span>';
            
            const engineBadge = task.tts_engine === 'fishspeech' ? 
                '<span class="badge badge-purple" style="background:rgba(139,92,246,0.2);color:#a78bfa">Fish</span>' : 
                '<span class="badge badge-green" style="background:rgba(16,185,129,0.2);color:#34d399">F5</span>';

            card.innerHTML = `
                <div class="card-header">
                    <span class="date">${dateStr}</span>
                    <div>
                        ${sepBadge}
                        ${engineBadge}
                        <span class="status-badge ${task.status}">${task.status}</span>
                    </div>
                </div>
                <div class="task-info">
                    <h3 title="${task.input_filename}">${task.input_filename || 'Unknown'}</h3>
                    <div class="message">${task.message || 'Initializing...'}</div>
                </div>
                <div class="task-actions">
                    ${renderActionButtons(task)}
                </div>
            `;
            list.appendChild(card);
        });

    } catch(e) {
        list.innerHTML = `<div class="error-msg">Failed: ${e.message}</div>`;
    }
}

function renderActionButtons(task) {
    if(!task || !task.task_id) return '';

    // Completed
    if (task.status === 'completed') {
        return `
            <button onclick="controlTask('${task.task_id}', 'delete')" class="ctrl-btn delete"><i class="fas fa-trash"></i></button>
            <a href="${task.output_file}" target="_blank" class="btn-download"><i class="fas fa-download"></i> Video</a>
            <button onclick="showProcessMonitor('${task.task_id}')" class="btn-monitor"><i class="fas fa-file-alt"></i></button>
        `;
    }
    
    // Failed
    if (task.status === 'failed') {
        return `
            <button onclick="controlTask('${task.task_id}', 'delete')" class="ctrl-btn delete"><i class="fas fa-trash"></i></button>
            <span class="error-txt">Failed</span>
        `;
    }

    // Active
    let btn = `<button onclick="controlTask('${task.task_id}', 'pause')" class="ctrl-btn pause"><i class="fas fa-pause"></i></button>`;
    if(task.status === 'paused') {
        btn = `<button onclick="controlTask('${task.task_id}', 'resume')" class="ctrl-btn resume"><i class="fas fa-play"></i></button>`;
    }

    return `
        <button onclick="showProcessMonitor('${task.task_id}')" class="btn-monitor">Monitor</button>
        ${btn}
        <button onclick="controlTask('${task.task_id}', 'cancel')" class="ctrl-btn stop"><i class="fas fa-stop"></i></button>
    `;
}

// -------------------------------------------------
// VIDEO TRANSLATION (Upload + YouTube)
// -------------------------------------------------
async function handleUpload(e) {
    e.preventDefault();
    
    // Use submitForm wrapper which handles FormData automatically
    submitForm(e.target, '/translate/upload', (data) => {
        if(data.task_id) showProcessMonitor(data.task_id);
    });
}

async function handleYouTube(e) {
    e.preventDefault();
    const formData = new FormData(e.target);
    
    // Build JSON payload for the YouTube endpoint
    const payload = {
        youtube_url: formData.get("url"),
        src_lang: formData.get("src_lang"),
        tgt_lang: formData.get("tgt_lang"),
        separate_audio: e.target.querySelector('[name="separate_audio"]').checked,
        tts_engine: formData.get("tts_engine") || "f5"
    };

    // Optional speaker name mapping (JSON string)
    const speakerNames = formData.get("speaker_names");
    if (speakerNames && speakerNames.trim().length > 0) {
        payload.speaker_names = speakerNames;
    }

    submitJson(e.target, '/translate/youtube', payload, (data) => {
        if(data.task_id) showProcessMonitor(data.task_id);
    });
}

async function reprocessTask(taskId) {
    const res = await fetch(`/api/tasks/${taskId}`);
    if (!res.ok) return alert('Task not found');

    // For brevity, we just call the server endpoint directly via fetch
    // (Implementation omitted – similar to handleUpload)
}

// -------------------------------------------------
// TRANSCRIBE TOOL
// -------------------------------------------------
async function handleTranscribe(e) {
    e.preventDefault();
    const resBox = document.getElementById('transcribe-result');
    resBox.style.display = 'none';

    submitForm(e.target, '/transcribe', (data) => {
        resBox.style.display = 'block';
        
        // Store result globally for download
        appState.transcriptionResult = data.result;

        // Populate Views
        document.getElementById('transcribe-orig').innerText = data.result.original_text;
        
        const transPre = document.getElementById('transcribe-trans');
        if (data.result.translated_text) {
            transPre.innerText = data.result.translated_text;
            // Show Translation Tab
            document.querySelector('button[onclick="showTab(\'transcribe-trans\')"]').style.display = 'inline-block';
        } else {
            transPre.innerText = "No translation requested.";
        }
    });
}

function downloadTranscript() {
    if (!appState.transcriptionResult) return;
    const blob = new Blob([JSON.stringify(appState.transcriptionResult, null, 2)], {type : 'application/json'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = "transcript.json";
    a.click();
}

// -------------------------------------------------
// TTS (Voice Clone) TOOL
// -------------------------------------------------
async function handleTTS(e) {
    e.preventDefault();
    const resBox = document.getElementById('tts-result');
    resBox.style.display = 'none';

    // tts/generate now returns a FileResponse (Blob)
    submitForm(e.target, '/tts/generate', (blob, isBlob) => {
        if (isBlob) {
            resBox.style.display = 'block';
            const url = URL.createObjectURL(blob);
            const audio = document.getElementById('tts-player');
            const dl = document.getElementById('tts-download');
            
            audio.src = url;
            audio.play();
            dl.href = url;
        }
    }, 'blob');
}

// -------------------------------------------------
// ANALYSIS TOOL
// -------------------------------------------------
async function handleAnalysis(e) {
    e.preventDefault();
    const resBox = document.getElementById('analysis-result');
    const content = document.getElementById('analysis-content');
    resBox.style.display = 'none';
    content.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing... (This uses LLM, please wait)';
    resBox.style.display = 'block';

    submitForm(e.target, '/analysis', (data) => {
        // Render Markdown
        if(window.marked) {
            content.innerHTML = marked.parse(data.analysis);
        } else {
            content.innerText = data.analysis;
        }
    });
}

// -------------------------------------------------
// RECORDER
// -------------------------------------------------
async function startRecording(mode) {
    try {
        // 1. Safety Check for Secure Context
        if (!navigator.mediaDevices && !navigator.mediaDevices.getUserMedia) {
            throw new Error("Browser blocked camera/mic access. Reason: Not a Secure Context (HTTPS). FIX: 1. Access via http://localhost:8000 (not IP address) 2. Or enable 'Insecure origins treated as secure' in chrome://flags");
        }

        let stream;
        const constraints = { audio: true, video: (mode !== 'audio') };

        if (mode === 'screen') {
            stream = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: true });
        } else {
            stream = await navigator.mediaDevices.getUserMedia(constraints);
        }
        
        const preview = document.getElementById('preview-video');
        if (mode !== 'audio') {
            preview.srcObject = stream;
            preview.style.display = 'block';
            preview.muted = true; // Prevent feedback loop
        } else {
            preview.style.display = 'none';
        }

        // Use correct MIME type for better compatibility
        const mimeType = MediaRecorder.isTypeSupported("video/webm; codecs=vp9") 
                        ? "video/webm; codecs=vp9" 
                        : "video/webm";

        appState.mediaRecorder = new MediaRecorder(stream, { mimeType });
        appState.recordChunks = [];

        appState.mediaRecorder.ondataavailable = e => {
            if (e.data.size > 0) appState.recordChunks.push(e.data);
        };

        appState.mediaRecorder.onstop = async () => {
            // Stop all tracks to turn off hardware light
            stream.getTracks().forEach(track => track.stop());

            // Determine ext
            const isAudio = (mode === 'audio');
            const ext = isAudio ? 'wav' : 'webm';
            const type = isAudio ? 'audio/wav' : 'video/webm';
            
            const blob = new Blob(appState.recordChunks, { type: type });
            
            // Upload
            const fd = new FormData();
            fd.append("file", blob, `recording_${Date.now}.${ext}`);
            
            const btnStop = document.querySelector('.btn-stop');
            btnStop.innerText = "Saving...";

            try {
                const res = await fetch('/recorder/save', { method:'POST', body:fd });
                if(!res.ok) throw new Error("Upload failed");
                
                const data = await res.json();
                appState.lastRecording = data;
                document.getElementById('last-recording').style.display = 'block';
                
            } catch(e) {
                alert("Save failed: " + e.message);
            } finally {
                btnStop.innerText = "Stop";
                btnStop.disabled = true;
                clearInterval(appState.timerInterval);
                document.getElementById('recording-indicator').style.display = 'none';
                if(preview.srcObject) preview.srcObject = null;
            }
        };

        appState.mediaRecorder.start();

        // UI Updates
        document.querySelector('.btn-stop').disabled = false;
        document.getElementById('recording-indicator').style.display = 'flex';
        startTimer();

    } catch(e) {
        alert("Recording Error: " + e.message);
        console.error(e);
    }
}

function stopRecording() {
    if (appState.mediaRecorder && appState.mediaRecorder.state !== 'inactive') {
        appState.mediaRecorder.stop();
    }
}

function startTimer() {
    let sec = 0;
    const timer = document.getElementById('timer');
    clearInterval(appState.timerInterval);
    appState.timerInterval = setInterval(() => {
        sec++;
        const m = Math.floor(sec/60).toString().padStart(2,'0');
        const s = (sec%60).toString().padStart(2,'0');
        timer.innerText = `${m}:${s}`;
    }, 1000);
}

function useRecording(targetView) {
    if (!appState.lastRecording) return;
    alert(`File saved on server at: ${appState.lastRecording.filename}\n\nPlease drag this file from your Uploads folder manually for now due to browser security limits.\n\n(Ideally, we would implement a Server File Selector in the other forms.)`);
}

// -------------------------------------------------
// HELPER: FORM SUBMISSION
// -------------------------------------------------
async function submitForm(form, url, callback, responseType='json') {
    const btn = form.querySelector('button[type="submit"]');
    const oldText = btn.innerText;
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';

    try {
        const fd = new FormData(form);
        
        const res = await fetch(url, {
            method: 'POST',
            body: fd
        });

        if (!res.ok) throw new Error("Server Error: " + res.statusText);

        let data;
        if (responseType === 'blob') {
            data = await res.blob();
            callback(data, true);
        } else {
            data = await res.json();
            callback(data, false);
        }

    } catch(e) {
        alert("Error: " + e.message);
    } finally {
        btn.disabled = false;
        btn.innerText = oldText;
    }
}

async function submitJson(form, url, payload, callback) {
    const btn = form.querySelector('button[type="submit"]');
    const oldText = btn.innerText; // Capture old text first
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';

    try {
        const res = await fetch(url, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        });

        if (!res.ok) throw new Error("Server Error: " + res.statusText);

        const data = await res.json();
        callback(data);

    } catch(e) {
        alert("Error: " + e.message);
    } finally {
        btn.disabled = false;
        btn.innerText = "Submit"; // Fallback text if capture failed, or could use oldText
    }
}

// -------------------------------------------------
// HELPER: WEBSOCKET MONITOR
// -------------------------------------------------
function connectWebSocket(taskId) {
   const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
   // Corrected WebSocket endpoint – matches backend route `/ws/{task_id}`
   appState.ws = new WebSocket(`${proto}://${window.location.host}/ws/${taskId}`);

   appState.ws.onmessage = (e) => {
       // Log every received WebSocket event for debugging / transparency
       console.log('WebSocket event received:', e.data);

       const msg = JSON.parse(e.data);
       if (msg.type === 'status_update') {
           updateMonitorUI(msg.data);
       } else if (msg.type === 'audio_chunk') {
           // New: Render playable chunk without autoplay
           const container = document.getElementById('audio-chunks');
           if (!container) return;
           const data = msg.data;
           try {
               const binary = atob(data.audio_base64);
               const len = binary.length;
               const bytes = new Uint8Array(len);
               for (let i = 0; i < len; i++) bytes[i] = binary.charCodeAt(i);
               const blob = new Blob([bytes.buffer], { type: 'audio/wav' });
               const url = URL.createObjectURL(blob);

               const wrapper = document.createElement('div');
               wrapper.style.marginBottom = '8px';

               const label = document.createElement('div');
               label.textContent = `Chunk ${data.start.toFixed(2)}s – ${data.end.toFixed(2)}s`;
               label.style.fontSize = '0.85rem';
               label.style.color = 'var(--text-muted)';

               const audio = document.createElement('audio');
               audio.controls = true;
               audio.src = url;

               wrapper.appendChild(label);
               wrapper.appendChild(audio);
               container.appendChild(wrapper);
           } catch (err) {
               console.error('Failed to render audio chunk:', err);
           }
       } else if (msg.type === 'speaker_samples') {
           // New: display speaker sample UI
           renderSpeakerSamples(msg.data);
       }
   };
}


 async function updateProcessUI(taskId) {
     const res = await fetch(`/api/tasks/${taskId}`);
     if (res.ok) {
         updateMonitorUI(await res.json());
     }
 }

function updateMonitorUI(data) {
    if (appState.activeView !== 'monitor' || appState.activeTaskId !== data.task_id) return;

    document.getElementById('monitor-filename').innerText = data.input_filename + " (" + (data.status || '...') + ")";
    document.getElementById('main-status').innerText = data.message || data.status;
    document.getElementById('progress-fill').style.width = (data.progress || 0) + "%";

    const ctrlDiv = document.getElementById('active-controls');
    
    if (data.status === 'completed') {
        ctrlDiv.innerHTML = `
            <a href="${data.output_file}" target="_blank" class="btn-download">Download Video</a>
            <button onclick="showDashboard()" class="btn-monitor">Back</button>
        `;
        // Load Script if available
        if (data.script_json) loadTranscript(data.script_json);
        // Show audio chunks container
        const chunkDiv = document.getElementById('audio-chunks');
        if (chunkDiv) chunkDiv.style.display = 'block';
    } else if (data.status === 'failed') {
        ctrlDiv.innerHTML = `<button onclick="controlTask('${data.task_id}', 'delete')" class="ctrl-btn delete">Delete</button>`;
    } else {
        // Active
        let pauseBtn = `<button onclick="controlTask('${data.task_id}', 'pause')" class="ctrl-btn pause">Pause</button>`;
        if (data.status === 'paused') {
            pauseBtn = `<button onclick="controlTask('${data.task_id}', 'resume')" class="ctrl-btn resume">Resume</button>`;
        }

        ctrlDiv.innerHTML = `
            ${pauseBtn}
            <button onclick="controlTask('${data.task_id}', 'cancel')" class="ctrl-btn stop">Cancel</button>
        `;
    }
}

async function loadTranscript(url) {
    const card = document.getElementById('transcript-card');
    const content = document.getElementById('transcript-content');
    card.style.display = 'block';
    
    document.getElementById('btn-download-json').href = url;

    try {
        const res = await fetch(url);
        appState.currentScript = await res.json();
        
        let html = '<table class="script-table"><tr><th>Time</th><th>Original</th><th>Translated</th></tr>';
        appState.currentScript.forEach(s => {
            html += `<tr><td>${s.start.toFixed(1)}s</td><td>${s.original}</td><td>${s.translated}</td></tr>`;
        });
        html += '</table>';
        content.innerHTML = html;
    } catch(e) {
        content.innerText = "Error loading transcript.";
    }
}

function copyTranscript() {
    if (!appState.currentScript) return;
    const txt = appState.currentScript.map(s => `[${s.start.toFixed(1)}s] ${s.original} -> ${s.translated}`).join('\n');
    navigator.clipboard.writeText(txt);
    alert("Copied!");
}

async function controlTask(id, action) {
    await fetch(`/api/tasks/${id}/control`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({action})
    });
    
    if(action === 'delete') refreshDashboard();
}

// -------------------------------------------------
// UI UTILS
// -------------------------------------------------
function showTab(tabId) {
    document.querySelectorAll('.code-block').forEach(el => el.style.display = 'none');
    document.getElementById(tabId).style.display = 'block';

    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    // Simple toggle logic for demo
    event.target.classList.add('active');
}

// Drag & Drop Visuals
document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('.file-input').forEach(input => {
        input.addEventListener('change', (e) => {
            const name = e.target.files[0]?.name;
            const span = e.target.parentElement.querySelector('.file-msg');
            if (span && name) span.innerText = name;
        });
    });
    
    // Restore last UI view from localStorage
    try {
        const savedView = localStorage.getItem('vdpu_active_view');
        if (savedView) {
            switchView(savedView);
        } else {
            showDashboard();
        }
    } catch (e) {
        console.warn('Unable to read UI state from localStorage:', e);
        showDashboard();
    }
});

// -------------------------------------------------
// NEW: SPEAKER NAMING UI
// -------------------------------------------------
function renderSpeakerSamples(speakerData) {
    // speakerData is an object: { "0": {audio_base64: "...", sample_rate: 24000, default_name: "...", reference_text: "..."}, ... }
    const container = document.getElementById('speaker-naming');
    if (!container) return;
    
    // Clear previous content
    container.innerHTML = '';
    
    const heading = document.createElement('h3');
    heading.innerText = 'Speaker Samples - Rename or Merge';
    container.appendChild(heading);
    
    const form = document.createElement('form');
    form.id = 'speaker-name-form';
    form.onsubmit = (e) => { e.preventDefault(); submitSpeakerNames(); };
    
    Object.entries(speakerData).forEach(([spkId, info]) => {
        const div = document.createElement('div');
        div.className = 'speaker-sample-item';
        div.style.marginBottom = '15px';
        
        const audio = document.createElement('audio');
        audio.controls = true;
        audio.src = `data:audio/wav;base64,${info.audio_base64}`;
        
        const label = document.createElement('label');
        label.innerText = `Speaker ${parseInt(spkId)+1} Name: `;
        label.style.marginRight = '8px';
        
        const input = document.createElement('input');
        input.type = 'text';
        input.name = `spk_${spkId}`;
        input.value = info.default_name;
        input.style.padding = '4px';
        input.style.width = '150px';
        
        const deleteBtn = document.createElement('button');
        deleteBtn.type = 'button';
        deleteBtn.className = 'btn-sm';
        deleteBtn.style.marginLeft = '8px';
        deleteBtn.innerText = 'Delete';
        deleteBtn.onclick = () => deleteSpeaker(spkId);
        
        div.appendChild(audio);
        div.appendChild(document.createElement('br'));
        div.appendChild(label);
        div.appendChild(input);
        div.appendChild(deleteBtn);
        
        form.appendChild(div);
    });
    
    const saveBtn = document.createElement('button');
    saveBtn.type = 'submit';
    saveBtn.className = 'btn-primary';
    saveBtn.innerText = 'Save Names';
    
    form.appendChild(saveBtn);
    container.appendChild(form);
    
    // Show the container
    container.style.display = 'block';
}

async function submitSpeakerNames() {
    if (!appState.activeTaskId) {
        alert('No active task.');
        return;
    }
    
    const form = document.getElementById('speaker-name-form');
    if (!form) return;
    
    const data = {};
    const inputs = form.querySelectorAll('input[name^="spk_"]');
    inputs.forEach(inp => {
        const spkId = inp.name.split('_')[1];
        data[spkId] = inp.value.trim() || `Speaker ${parseInt(spkId)+1}`;
    });
    
    try {
        const res = await fetch(`/api/tasks/${appState.activeTaskId}/speakers`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({names: data})
        });
        if (!res.ok) throw new Error('Failed to save speaker names');
        const resp = await res.json();
        alert('Speaker names saved.');
        // Hide naming UI
        document.getElementById('speaker-naming').style.display = 'none';
        // Update local cache so future chunks use the new names
        appState.speakerNames = data;
    } catch (e) {
        alert('Error: ' + e.message);
    }
}

async function deleteSpeaker(speakerId) {
    if (!appState.activeTaskId) {
        alert('No active task.');
        return;
    }
    if (!confirm(`Delete speaker ${parseInt(speakerId)+1}? This will keep existing audio unchanged.`)) return;
    
    try {
        const res = await fetch(`/api/tasks/${appState.activeTaskId}/speakers/${speakerId}`, {
            method: 'DELETE'
        });
        if (!res.ok) throw new Error('Failed to delete speaker');
        const resp = await res.json();
        alert(resp.message);
        // Remove speaker UI element
        const form = document.getElementById('speaker-name-form');
        const elem = form.querySelector(`input[name="spk_${speakerId}"]`).parentElement;
        elem.remove();
    } catch (e) {
        alert('Error: ' + e.message);
    }
}

// Expose a helper for monitor view to open the naming UI manually (optional)
function openSpeakerNaming() {
    const container = document.getElementById('speaker-naming');
    if (container) container.style.display = 'block';
}
