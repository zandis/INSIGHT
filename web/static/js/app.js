/**
 * INSIGHT Web Application JavaScript
 * Core functionality for the research platform
 */

// =============================================================================
// Authentication
// =============================================================================

const AUTH_TOKEN_KEY = 'insight_token';
const AUTH_USER_KEY = 'insight_user';

function getAuthToken() {
    return localStorage.getItem(AUTH_TOKEN_KEY);
}

function setAuthToken(token) {
    localStorage.setItem(AUTH_TOKEN_KEY, token);
}

function getAuthUser() {
    const user = localStorage.getItem(AUTH_USER_KEY);
    return user ? JSON.parse(user) : null;
}

function setAuthUser(user) {
    localStorage.setItem(AUTH_USER_KEY, JSON.stringify(user));
}

function getAuthHeaders() {
    const token = getAuthToken();
    return token ? { 'Authorization': `Bearer ${token}` } : {};
}

function isAuthenticated() {
    return !!getAuthToken();
}

function logout() {
    localStorage.removeItem(AUTH_TOKEN_KEY);
    localStorage.removeItem(AUTH_USER_KEY);
    window.location.href = '/login';
}

// =============================================================================
// Toast Notifications
// =============================================================================

function showToast(message, type = 'info', duration = 5000) {
    const container = document.getElementById('toastContainer');
    if (!container) return;

    const toastId = 'toast-' + Date.now();
    const iconMap = {
        'success': 'check-circle-fill',
        'danger': 'exclamation-triangle-fill',
        'warning': 'exclamation-circle-fill',
        'info': 'info-circle-fill'
    };

    const toast = document.createElement('div');
    toast.id = toastId;
    toast.className = `toast align-items-center text-bg-${type} border-0`;
    toast.setAttribute('role', 'alert');
    toast.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">
                <i class="bi bi-${iconMap[type] || 'info-circle-fill'} me-2"></i>
                ${escapeHtml(message)}
            </div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
        </div>
    `;

    container.appendChild(toast);

    const bsToast = new bootstrap.Toast(toast, { delay: duration });
    bsToast.show();

    toast.addEventListener('hidden.bs.toast', () => toast.remove());
}

// =============================================================================
// Utility Functions
// =============================================================================

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatDate(dateString) {
    if (!dateString) return '-';
    const date = new Date(dateString);
    return date.toLocaleString();
}

function formatRelativeTime(dateString) {
    if (!dateString) return '-';
    const date = new Date(dateString);
    const now = new Date();
    const diff = Math.floor((now - date) / 1000);

    if (diff < 60) return 'just now';
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    if (diff < 604800) return `${Math.floor(diff / 86400)}d ago`;
    return date.toLocaleDateString();
}

function getStatusColor(status) {
    const colors = {
        'pending': 'secondary',
        'running': 'warning',
        'completed': 'success',
        'failed': 'danger',
        'cancelled': 'dark'
    };
    return colors[status] || 'secondary';
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function throttle(func, limit) {
    let inThrottle;
    return function(...args) {
        if (!inThrottle) {
            func.apply(this, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

// =============================================================================
// API Helpers
// =============================================================================

async function apiRequest(url, options = {}) {
    const defaultOptions = {
        headers: {
            'Content-Type': 'application/json',
            ...getAuthHeaders()
        }
    };

    const response = await fetch(url, { ...defaultOptions, ...options });

    if (response.status === 401) {
        logout();
        return null;
    }

    return response;
}

async function apiGet(url) {
    return apiRequest(url);
}

async function apiPost(url, data) {
    return apiRequest(url, {
        method: 'POST',
        body: JSON.stringify(data)
    });
}

async function apiPut(url, data) {
    return apiRequest(url, {
        method: 'PUT',
        body: JSON.stringify(data)
    });
}

async function apiDelete(url) {
    return apiRequest(url, {
        method: 'DELETE'
    });
}

// =============================================================================
// Notifications
// =============================================================================

async function loadNotifications() {
    try {
        const response = await apiGet('/api/notifications');
        if (response && response.ok) {
            const notifications = await response.json();
            updateNotificationsBadge(notifications);
            updateNotificationsList(notifications);
        }
    } catch (error) {
        console.error('Failed to load notifications:', error);
    }
}

function updateNotificationsBadge(notifications) {
    const unread = notifications.filter(n => !n.read).length;
    const badge = document.getElementById('notificationCount');
    if (badge) {
        badge.textContent = unread;
        badge.style.display = unread > 0 ? 'inline-block' : 'none';
    }
}

function updateNotificationsList(notifications) {
    const list = document.getElementById('notificationsList');
    if (!list) return;

    if (notifications.length === 0) {
        list.innerHTML = '<li><span class="dropdown-item-text text-muted">No notifications</span></li>';
        return;
    }

    list.innerHTML = notifications.slice(0, 10).map(n => `
        <li>
            <a class="dropdown-item ${n.read ? '' : 'fw-bold'}" href="#" onclick="handleNotification('${n.id}')">
                <div class="d-flex justify-content-between">
                    <span>${escapeHtml(n.title)}</span>
                    <small class="text-muted">${formatRelativeTime(n.created_at)}</small>
                </div>
                <small class="text-muted">${escapeHtml(n.message.substring(0, 50))}...</small>
            </a>
        </li>
    `).join('') + `
        <li><hr class="dropdown-divider"></li>
        <li><a class="dropdown-item text-center" href="/notifications">View All</a></li>
    `;
}

async function handleNotification(id) {
    await apiPost(`/api/notifications/${id}/read`);
    loadNotifications();
}

// =============================================================================
// User Session
// =============================================================================

function updateUserUI() {
    const user = getAuthUser();
    const userName = document.getElementById('userName');
    const adminLink = document.getElementById('adminLink');

    if (userName && user) {
        userName.textContent = user.username || 'User';
    }

    if (adminLink && user && user.is_admin) {
        adminLink.style.display = 'block';
    }
}

// =============================================================================
// WebSocket Connection
// =============================================================================

class WebSocketManager {
    constructor(url) {
        this.url = url;
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
        this.handlers = {};
    }

    connect() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        this.ws = new WebSocket(`${protocol}//${window.location.host}${this.url}`);

        this.ws.onopen = () => {
            console.log('WebSocket connected');
            this.reconnectAttempts = 0;
            this.emit('open');
        };

        this.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.emit('message', data);
                if (data.type) {
                    this.emit(data.type, data);
                }
            } catch (e) {
                console.error('Failed to parse WebSocket message:', e);
            }
        };

        this.ws.onclose = () => {
            console.log('WebSocket disconnected');
            this.emit('close');
            this.reconnect();
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.emit('error', error);
        };
    }

    reconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
            console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);
            setTimeout(() => this.connect(), delay);
        }
    }

    on(event, handler) {
        if (!this.handlers[event]) {
            this.handlers[event] = [];
        }
        this.handlers[event].push(handler);
    }

    off(event, handler) {
        if (this.handlers[event]) {
            this.handlers[event] = this.handlers[event].filter(h => h !== handler);
        }
    }

    emit(event, data) {
        if (this.handlers[event]) {
            this.handlers[event].forEach(handler => handler(data));
        }
    }

    send(data) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(data));
        }
    }

    close() {
        if (this.ws) {
            this.ws.close();
        }
    }
}

// =============================================================================
// Charts Helper
// =============================================================================

function createLineChart(canvasId, labels, data, options = {}) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;

    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                borderColor: options.borderColor || 'rgb(75, 192, 192)',
                backgroundColor: options.backgroundColor || 'rgba(75, 192, 192, 0.1)',
                tension: 0.1,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: { beginAtZero: true }
            },
            ...options.chartOptions
        }
    });
}

function createDoughnutChart(canvasId, labels, data, colors) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;

    return new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: colors
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'bottom' }
            }
        }
    });
}

// =============================================================================
// Form Validation
// =============================================================================

function validateForm(formElement) {
    const inputs = formElement.querySelectorAll('input[required], textarea[required], select[required]');
    let isValid = true;

    inputs.forEach(input => {
        if (!input.value.trim()) {
            input.classList.add('is-invalid');
            isValid = false;
        } else {
            input.classList.remove('is-invalid');
        }
    });

    return isValid;
}

function clearFormValidation(formElement) {
    const inputs = formElement.querySelectorAll('.is-invalid');
    inputs.forEach(input => input.classList.remove('is-invalid'));
}

// =============================================================================
// Loading States
// =============================================================================

function showLoading(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.innerHTML = `
            <div class="d-flex justify-content-center py-5">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
            </div>
        `;
    }
}

function hideLoading(elementId, content) {
    const element = document.getElementById(elementId);
    if (element) {
        element.innerHTML = content;
    }
}

// =============================================================================
// Local Storage Helpers
// =============================================================================

function saveToStorage(key, value) {
    try {
        localStorage.setItem(key, JSON.stringify(value));
    } catch (e) {
        console.error('Failed to save to localStorage:', e);
    }
}

function loadFromStorage(key, defaultValue = null) {
    try {
        const item = localStorage.getItem(key);
        return item ? JSON.parse(item) : defaultValue;
    } catch (e) {
        console.error('Failed to load from localStorage:', e);
        return defaultValue;
    }
}

// =============================================================================
// Keyboard Shortcuts
// =============================================================================

document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + K: Focus search
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        const searchInput = document.getElementById('searchInput') ||
                           document.querySelector('input[type="search"]');
        if (searchInput) searchInput.focus();
    }

    // Ctrl/Cmd + N: New research
    if ((e.ctrlKey || e.metaKey) && e.key === 'n') {
        if (!e.target.matches('input, textarea')) {
            e.preventDefault();
            window.location.href = '/research/new';
        }
    }

    // Escape: Close modals
    if (e.key === 'Escape') {
        const modals = document.querySelectorAll('.modal.show');
        modals.forEach(modal => {
            bootstrap.Modal.getInstance(modal)?.hide();
        });
    }
});

// =============================================================================
// Initialization
// =============================================================================

document.addEventListener('DOMContentLoaded', function() {
    // Update user UI
    updateUserUI();

    // Load notifications if authenticated
    if (isAuthenticated()) {
        loadNotifications();
        setInterval(loadNotifications, 60000); // Refresh every minute
    }

    // Logout button
    const logoutBtn = document.getElementById('logoutBtn');
    if (logoutBtn) {
        logoutBtn.addEventListener('click', function(e) {
            e.preventDefault();
            logout();
        });
    }

    // Initialize tooltips
    const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    tooltipTriggerList.forEach(el => new bootstrap.Tooltip(el));

    // Initialize popovers
    const popoverTriggerList = document.querySelectorAll('[data-bs-toggle="popover"]');
    popoverTriggerList.forEach(el => new bootstrap.Popover(el));

    // Form validation
    const forms = document.querySelectorAll('form[data-validate]');
    forms.forEach(form => {
        form.addEventListener('submit', function(e) {
            if (!validateForm(form)) {
                e.preventDefault();
                e.stopPropagation();
            }
        });
    });

    console.log('INSIGHT Web Application initialized');
});

// =============================================================================
// Export for use in other scripts
// =============================================================================

window.INSIGHT = {
    api: { get: apiGet, post: apiPost, put: apiPut, delete: apiDelete },
    auth: { getToken: getAuthToken, setToken: setAuthToken, isAuthenticated, logout },
    toast: showToast,
    utils: { escapeHtml, formatDate, formatRelativeTime, getStatusColor, debounce, throttle },
    WebSocketManager
};
