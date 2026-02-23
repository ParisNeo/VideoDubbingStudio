// Main Application Entry Point
import * as Dashboard from './views/dashboard.js';
import * as VideoDub from './views/video_dub.js';
import * as Transcribe from './views/transcribe.js';
import * as Analysis from './views/analysis.js';
import * as Recorder from './views/recorder.js';

// Global state - Initialize immediately to avoid undefined errors in modules
window.appState = {
    currentTaskId: null,
    currentView: null // Start with null so first switch always triggers
};

const modules = {
    'dashboard': Dashboard,
    'video_dub': VideoDub,
    'transcribe': Transcribe,
    'analysis': Analysis,
    'recorder': Recorder
};

document.addEventListener('DOMContentLoaded', () => {
    console.log("%c VoiceDub Pro - Initializing ", 
                "background: #222; color: #bada55; font-size: 14px");

    // Initialize all modules
    Object.entries(modules).forEach(([name, mod]) => {
        try {
            if (mod && mod.init) {
                mod.init();
                console.log(`✅ ${name} initialized`);
            }
        } catch (e) {
            console.error(`❌ ${name} failed:`, e);
        }
    });

    // Listen for project creation events from the modal
    window.addEventListener('createProject', (e) => {
        const projectType = e.detail.type;
        handleProjectCreation(projectType);
    });

    // Always start with dashboard on fresh load
    doSwitchView('dashboard');
});

function handleProjectCreation(type) {
    const mod = modules[type];
    if (!mod) return;

    doSwitchView(type);
    if (mod.startNewProject) mod.startNewProject();
}

// Main switch view function
export function doSwitchView(viewName) {
    // Validate view exists
    if (!modules[viewName]) return;

    const target = document.getElementById(`view-${viewName}`);
    if (!target) return;

    const currentId = window.appState.currentView;
    
    // If target is same as current and already visible, do nothing
    if (currentId === viewName && target.classList.contains('active')) {
        return;
    }

    // Hide all views first to ensure a clean state
    document.querySelectorAll('.view-section').forEach(el => {
        el.style.display = 'none';
        el.classList.remove('active');
    });

    // Show target view
    target.style.display = 'block';
    target.classList.add('active');

    // Update global state
    window.appState.currentView = viewName;

    // Save to localStorage for in-app navigation persistence
    try {
        localStorage.setItem('vdpu_active_view', viewName);
    } catch (e) {
        console.warn('Could not save view preference:', e);
    }

    // Trigger module's onShow if exists
    const mod = modules[viewName];
    if (mod && mod.onShow) {
        console.log(`Calling onShow for ${viewName}`);
        try {
            mod.onShow();
        } catch (e) {
            console.error(`onShow failed for ${viewName}:`, e);
        }
    }

    console.log(`Switched to view: ${viewName}`);
}

// Back to dashboard function for modules
export function backToDashboard() {
    doSwitchView('dashboard');
}

// Global UI Dialog overrides (replacing native alert/confirm)
window.uiAlert = function(message, title = "Notice") {
    return new Promise(resolve => {
        const overlay = document.getElementById('ui-dialog-overlay');
        if (!overlay) { alert(message); resolve(); return; } // Fallback
        
        document.getElementById('ui-dialog-title').innerText = title;
        document.getElementById('ui-dialog-message').innerText = message;
        
        const cancelBtn = document.getElementById('ui-dialog-cancel');
        const okBtn = document.getElementById('ui-dialog-ok');
        
        cancelBtn.style.display = 'none';
        
        okBtn.onclick = () => {
            overlay.style.display = 'none';
            resolve();
        };
        
        overlay.style.display = 'flex';
    });
};

window.uiConfirm = function(message, title = "Confirm Action") {
    return new Promise(resolve => {
        const overlay = document.getElementById('ui-dialog-overlay');
        if (!overlay) { resolve(confirm(message)); return; } // Fallback
        
        document.getElementById('ui-dialog-title').innerText = title;
        document.getElementById('ui-dialog-message').innerText = message;
        
        const cancelBtn = document.getElementById('ui-dialog-cancel');
        const okBtn = document.getElementById('ui-dialog-ok');
        
        cancelBtn.style.display = 'block';
        
        cancelBtn.onclick = () => {
            overlay.style.display = 'none';
            resolve(false);
        };
        okBtn.onclick = () => {
            overlay.style.display = 'none';
            resolve(true);
        };
        
        overlay.style.display = 'flex';
    });
};

// Bind to window for inline handlers
window.doSwitchView = doSwitchView;
window.backToDashboard = backToDashboard;
