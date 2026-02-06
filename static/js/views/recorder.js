import { formatTime } from '../api.js';

let mediaRecorder;
let audioChunks = [];
let startTime;
let timerInterval;

export function init() {
    const startBtn = document.getElementById('rec-start-btn');
    const stopBtn = document.getElementById('rec-stop-btn');

    if (!startBtn || !stopBtn) {
        console.warn('Recorder buttons not found');
        return;
    }

    startBtn.addEventListener('click', async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];

            mediaRecorder.ondataavailable = event => {
                if (event.data.size > 0) audioChunks.push(event.data);
            };
            
            mediaRecorder.onstop = saveRecording;

            mediaRecorder.start();
            startBtn.disabled = true;
            stopBtn.disabled = false;
            startTimer();
        } catch (err) {
            alert('Could not access microphone: ' + err.message);
        }
    });

    stopBtn.addEventListener('click', () => {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
        }
        startBtn.disabled = false;
        stopBtn.disabled = true;
        stopTimer();
    });
}

function startTimer() {
    startTime = Date.now();
    const display = document.getElementById('timer-display');
    timerInterval = setInterval(() => {
        const diff = Math.floor((Date.now() - startTime) / 1000);
        if (display) display.textContent = formatTime(diff);
    }, 1000);
}

function stopTimer() {
    clearInterval(timerInterval);
}

async function saveRecording() {
    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
    const formData = new FormData();
    formData.append('file', audioBlob, 'recording.wav');

    const list = document.getElementById('recordings-list');
    const item = document.createElement('div');
    item.className = 'recording-item';
    item.textContent = "Uploading...";
    if (list) list.appendChild(item);

    try {
        const res = await fetch('/recorder/save', { 
            method: 'POST', 
            body: formData 
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        item.innerHTML = `
            <span>${data.filename}</span>
            <audio controls src="/uploads/${data.filename}"></audio>
        `;
    } catch (e) {
        item.textContent = "Upload failed: " + e.message;
    }
}
