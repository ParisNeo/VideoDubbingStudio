// Main Application Entry Point
import * as Dashboard from './views/dashboard.js';
import * as VideoDub from './views/video_dub.js';
import * as Transcribe from './views/transcribe.js';
import * as Analysis from './views/analysis.js';
import * as Recorder from './views/recorder.js';

const modules = {
    'dashboard': Dashboard,
    'video_dub': VideoDub,
    'transcribe': Transcribe,
    'analysis': Analysis,
    'recorder': Recorder
};

// Global state
window.appState = {
    currentTaskId: null,
    currentView: 'dashboard'
};

document.addEventListener('DOMContentLoaded', () => {
    console.log("%c VoiceDub Pro - Initializing ", 
                "background: #222; color: #bada55; font-size: 14px");

    // Initialize all modules
    Object.entries(modules).forEach(([name, mod]) => {
        try {
            if (mod.init) {
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
        console.log(`Creating project of type: ${projectType}`);
        handleProjectCreation(projectType);
    });

    // Always start with dashboard on fresh load
    // User's saved preference is only used for in-app navigation, not initial load
    setTimeout(() => {
        doSwitchView('dashboard');
    }, 100);
});

function handleProjectCreation(type) {
    const mod = modules[type];
    if (!mod) {
        console.error(`Unknown project type: ${type}`);
        return;
    }

    // Switch to the appropriate view
    doSwitchView(type);
    
    // If the module has a new project function, call it
    if (mod.startNewProject) {
        mod.startNewProject();
    }
}

// Main switch view function
export function doSwitchView(viewName) {
    console.log(`Switching to view: ${viewName}`);
    
    // Validate view exists
    if (!modules[viewName]) {
        console.error(`Unknown view: ${viewName}`);
        return;
    }

    // Get current and target elements
    const currentId = window.appState.currentView;
    const target = document.getElementById(`view-${viewName}`);
    
    if (!target) {
        console.error(`View element not found: view-${viewName}`);
        return;
    }

    // Hide current view if exists
    if (currentId) {
        const currentEl = document.getElementById(`view-${currentId}`);
        if (currentEl) {
            currentEl.style.display = 'none';
            currentEl.classList.remove('active');
        }
    }

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

// Bind to window for inline handlers
window.doSwitchView = doSwitchView;
window.backToDashboard = backToDashboard;
