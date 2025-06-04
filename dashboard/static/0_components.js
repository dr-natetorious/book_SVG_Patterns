/**
 * Release Dashboard Component
 * Renders and manages release workflow cards with real-time SSE updates
 */
class ReleaseDashboard {
    constructor(containerSelector = '#changesGrid', sseEndpoint = '/stream') {
        this.container = document.querySelector(containerSelector);
        this.sseEndpoint = sseEndpoint;
        this.eventSource = null;
        this.changes = new Map();
        this.currentFilter = 'all';
        this.tooltips = new Map();
        
        // Bind methods to preserve 'this' context
        this.handleSSEMessage = this.handleSSEMessage.bind(this);
        this.handleSSEError = this.handleSSEError.bind(this);
        this.handleFilterChange = this.handleFilterChange.bind(this);
        
        this.init();
    }
    
    init() {
        this.setupEventSource();
        this.setupFilters();
        this.setupModalHandlers();
    }
    
    // SSE Connection Management
    setupEventSource() {
        if (this.eventSource) {
            this.eventSource.close();
        }
        
        this.eventSource = new EventSource(this.sseEndpoint);
        this.eventSource.onmessage = this.handleSSEMessage;
        this.eventSource.onerror = this.handleSSEError;
        
        this.eventSource.onopen = () => {
            console.log('SSE connection established');
            this.updateConnectionStatus('Connected');
        };
    }
    
    handleSSEMessage(event) {
        try {
            const data = JSON.parse(event.data);
            
            switch (data.type) {
                case 'initial_data':
                    this.handleInitialData(data.changes);
                    break;
                case 'change_update':
                    this.handleChangeUpdate(data.change);
                    break;
                case 'broadcast':
                    console.log('Broadcast:', data.message);
                    break;
                case 'heartbeat':
                    this.updateLastUpdated();
                    break;
                default:
                    console.log('SSE message:', data);
            }
        } catch (error) {
            console.error('Error parsing SSE message:', error);
        }
    }
    
    handleSSEError(error) {
        console.error('SSE connection error:', error);
        this.updateConnectionStatus('Disconnected');
        
        // Reconnect after 5 seconds
        setTimeout(() => {
            console.log('Attempting to reconnect...');
            this.setupEventSource();
        }, 5000);
    }
    
    // Data Handling
    handleInitialData(changes) {
        this.changes.clear();
        changes.forEach(change => {
            this.changes.set(change.id, change);
        });
        this.renderAllCards();
        this.updateLastUpdated();
    }
    
    handleChangeUpdate(change) {
        this.changes.set(change.id, change);
        this.renderCard(change);
        this.updateLastUpdated();
    }
    
    // Card Rendering
    renderAllCards() {
        this.container.innerHTML = '';
        this.changes.forEach(change => {
            this.renderCard(change, false);
        });
        this.applyCurrentFilter();
        this.initializeTooltips();
    }
    
    renderCard(change, updateOnly = true) {
        const cardHtml = this.generateCardHtml(change);
        
        if (updateOnly) {
            const existingCard = document.querySelector(`[data-change-id="${change.id}"]`);
            if (existingCard) {
                const wrapper = existingCard.closest('.col-xl-3');
                wrapper.innerHTML = cardHtml;
                this.initializeCardTooltips(wrapper);
                return;
            }
        }
        
        // Create new card
        const wrapper = document.createElement('div');
        wrapper.className = 'col-xl-3 col-lg-4 col-md-6';
        wrapper.innerHTML = cardHtml;
        this.container.appendChild(wrapper);
        
        this.initializeCardTooltips(wrapper);
        this.applyCurrentFilter();
    }
    
    generateCardHtml(change) {
        return `
            <div class="card release-card ${this.getCardClasses(change)}" 
                 role="article" 
                 aria-labelledby="change-${change.id}-title"
                 data-change-id="${change.id}"
                 data-role="${this.getDataRoles(change)}">
                
                ${this.generateCardHeader(change)}
                ${this.generateCardBody(change)}
                ${this.generateCardFooter(change)}
            </div>
        `;
    }
    
    generateCardHeader(change) {
        const riskConfig = this.getRiskConfig(change.risk);
        const statusConfig = this.getStatusConfig(change);
        
        return `
            <div class="card-header">
                <div class="d-flex align-items-center justify-content-between">
                    <div class="d-flex align-items-center">
                        <div class="risk-indicator ${riskConfig.class}" 
                             role="img" 
                             aria-label="${riskConfig.label}"
                             title="${riskConfig.title}">${riskConfig.icon}</div>
                        <div>
                            <div class="fw-bold text-dark" style="font-size: 0.9rem;" id="change-${change.id}-title">${change.id}</div>
                            <div style="font-size: 0.8rem; color: #6c757d;">${change.services.join(', ')}</div>
                        </div>
                    </div>
                    <div class="maintenance-status ${statusConfig.class}" 
                         aria-label="${statusConfig.label}">
                        <i class="bi ${statusConfig.icon}" aria-hidden="true"></i> 
                        <span>${statusConfig.text}</span>
                    </div>
                </div>
            </div>
        `;
    }
    
    generateCardBody(change) {
        let bodyContent = this.generateCurrentStatus(change);
        
        if (change.blockers && change.blockers.length > 0) {
            bodyContent += this.generateBlockersSection(change.blockers);
        }
        
        bodyContent += this.generateImpactIndicators(change);
        bodyContent += `<div class="scroll-hint" id="scrollHint${change.id}">
            <i class="bi bi-chevron-double-down" aria-hidden="true"></i> Scroll
        </div>`;
        
        return `<div class="card-body">${bodyContent}</div>`;
    }
    
    generateCurrentStatus(change) {
        const statusConfig = this.getDetailedStatusConfig(change);
        
        let progressBar = '';
        if (change.progress !== undefined) {
            const progressClass = change.status === 'rollback' ? 'bg-danger' : 'bg-primary';
            progressBar = `
                <div class="progress mb-2" style="height: 6px;" role="progressbar" 
                     aria-valuenow="${change.progress}" aria-valuemin="0" aria-valuemax="100">
                    <div class="progress-bar ${progressClass} progress-bar-striped progress-bar-animated" 
                         style="width: ${change.progress}%"></div>
                </div>
            `;
        }
        
        const nextActionClass = change.urgent ? 'urgent' : '';
        
        return `
            <div class="current-status ${statusConfig.class}">
                <div class="status-title">
                    <i class="bi ${statusConfig.icon}" aria-hidden="true"></i>
                    <span>${statusConfig.title}</span>
                    ${statusConfig.showSpinner ? `
                        <div class="spinner-border spinner-border-sm ${statusConfig.spinnerClass}" 
                             role="status" aria-label="${statusConfig.spinnerLabel}">
                            <span class="sr-only">Loading...</span>
                        </div>
                    ` : ''}
                </div>
                <div class="status-description">${statusConfig.description}</div>
                ${progressBar}
                <div class="next-action ${nextActionClass}">
                    <i class="bi ${statusConfig.actionIcon}" aria-hidden="true"></i> ${statusConfig.actionText}
                </div>
            </div>
        `;
    }
    
    generateBlockersSection(blockers) {
        const blockerItems = blockers.map(blocker => `
            <div class="blocker-item">
                <i class="bi bi-exclamation-triangle text-warning" aria-hidden="true"></i>
                <span>${blocker}</span>
            </div>
        `).join('');
        
        return `
            <div class="blockers-section">
                <div class="fw-bold text-warning mb-2">
                    <i class="bi bi-exclamation-triangle" aria-hidden="true"></i> Blockers
                </div>
                ${blockerItems}
            </div>
        `;
    }
    
    generateImpactIndicators(change) {
        const indicators = [];
        
        // Services impact
        if (change.services.length > 0) {
            const serviceCount = change.services.length;
            const serviceIcon = serviceCount === 1 ? 'bi-server' : 'bi-servers';
            indicators.push(`
                <span class="impact-badge impact-services">
                    <i class="bi ${serviceIcon}" aria-hidden="true"></i> ${serviceCount} Service${serviceCount > 1 ? 's' : ''}
                </span>
            `);
        }
        
        // Risk-based indicators
        if (change.risk === 'high' || change.risk === 'critical') {
            indicators.push(`
                <span class="impact-badge impact-downtime">
                    <i class="bi bi-exclamation-triangle" aria-hidden="true"></i> High Impact
                </span>
            `);
        }
        
        // User impact (mock - would come from actual data)
        const userCount = this.estimateUserImpact(change);
        if (userCount) {
            indicators.push(`
                <span class="impact-badge impact-users">
                    <i class="bi bi-people" aria-hidden="true"></i> ${userCount}
                </span>
            `);
        }
        
        return `<div class="impact-indicators">${indicators.join('')}</div>`;
    }
    
    generateCardFooter(change) {
        const avatarInitials = this.getInitials(change.assignee);
        const avatarColor = this.getAvatarColor(change.assignee);
        
        return `
            <div class="card-footer">
                <div class="d-flex justify-content-between align-items-center">
                    <div class="assignee-info">
                        <div class="assignee-avatar" style="background: ${avatarColor};" title="${change.assignee}">${avatarInitials}</div>
                        <span>${change.assignee}</span>
                        <span class="time-estimate">${change.eta || 'No ETA'}</span>
                    </div>
                    <button class="btn btn-outline-primary btn-sm" 
                            onclick="dashboard.showWorkflow('${change.id}')"
                            aria-label="View workflow diagram for ${change.id}">
                        <i class="bi bi-diagram-3" aria-hidden="true"></i> Workflow
                    </button>
                </div>
            </div>
        `;
    }
    
    // Configuration Methods
    getRiskConfig(risk) {
        const configs = {
            low: { class: 'risk-low', icon: 'L', label: 'Low risk change', title: 'Low Risk - Auto-approved' },
            medium: { class: 'risk-medium', icon: 'M', label: 'Medium risk change with striped pattern', title: 'Medium Risk - Manager approval required' },
            high: { class: 'risk-high', icon: 'H', label: 'High risk change with diagonal stripes', title: 'High Risk - Director approval required' },
            critical: { class: 'risk-critical', icon: 'C', label: 'Critical risk change with cross-hatch pattern', title: 'Critical Risk - Emergency only' }
        };
        return configs[risk] || configs.low;
    }
    
    getStatusConfig(change) {
        const configs = {
            deploying: { class: 'maintenance-active', icon: 'bi-gear-fill', text: change.maintenance_window || 'Deploying', label: 'Currently deploying' },
            rollback: { class: 'maintenance-active', icon: 'bi-arrow-counterclockwise', text: 'Rollback', label: 'Emergency rollback in progress' },
            blocked: { class: 'maintenance-blocked', icon: 'bi-exclamation-triangle', text: 'Blocked', label: 'Blocked - missing requirements' },
            ready: { class: 'maintenance-pending', icon: 'bi-check-circle', text: 'Ready', label: 'Ready for deployment' },
            complete: { class: 'maintenance-complete', icon: 'bi-check-circle-fill', text: 'Complete', label: 'Deployment complete' }
        };
        return configs[change.status] || { class: 'maintenance-pending', icon: 'bi-clock', text: 'Pending', label: 'Status pending' };
    }
    
    getDetailedStatusConfig(change) {
        const configs = {
            deploying: {
                class: 'running',
                icon: 'bi-gear-fill text-primary',
                title: `Deploying - Step ${change.current_step || 1} of ${change.total_steps || 5}`,
                description: `Deployment in progress - ETA ${change.eta || 'Unknown'}`,
                actionIcon: 'bi-clock',
                actionText: 'Monitoring deployment progress',
                showSpinner: true,
                spinnerClass: 'text-primary',
                spinnerLabel: 'Deployment in progress'
            },
            rollback: {
                class: 'running',
                icon: 'bi-arrow-counterclockwise text-danger',
                title: `Rolling Back - Step ${change.current_step || 1} of ${change.total_steps || 3}`,
                description: 'Emergency rollback in progress - Critical issue detected',
                actionIcon: 'bi-exclamation-triangle',
                actionText: `Monitor rollback completion - ETA ${change.eta || 'Unknown'}`,
                showSpinner: true,
                spinnerClass: 'text-danger',
                spinnerLabel: 'Rollback in progress'
            },
            blocked: {
                class: 'blocked',
                icon: 'bi-pause-circle text-warning',
                title: 'Blocked - Waiting for Requirements',
                description: 'Cannot proceed until blockers are resolved',
                actionIcon: 'bi-person-check',
                actionText: 'Resolve blocking issues to continue'
            },
            ready: {
                class: '',
                icon: 'bi-check-circle-fill text-success',
                title: 'Ready for Deployment',
                description: 'All requirements met, ready to deploy',
                actionIcon: 'bi-calendar-event',
                actionText: `Deploy in ${change.maintenance_window || 'next window'}`
            },
            complete: {
                class: '',
                icon: 'bi-check-circle-fill text-success',
                title: 'Deployment Complete',
                description: 'Successfully deployed and verified',
                actionIcon: 'bi-check-circle',
                actionText: 'Deployment successful'
            }
        };
        
        return configs[change.status] || {
            class: '',
            icon: 'bi-clock text-muted',
            title: 'Status Unknown',
            description: 'Change status not recognized',
            actionIcon: 'bi-question-circle',
            actionText: 'Check change status'
        };
    }
    
    getCardClasses(change) {
        const classes = [];
        
        // Status patterns for accessibility
        classes.push(`status-pattern-${change.status}`);
        
        // Special states
        if (change.is_my_change) classes.push('my-change');
        if (change.status === 'deploying' || change.status === 'rollback') classes.push('running');
        if (change.status === 'blocked') classes.push('blocked');
        if (change.urgent) classes.push('urgent');
        
        return classes.join(' ');
    }
    
    getDataRoles(change) {
        const roles = [];
        if (change.is_my_change) roles.push('my');
        if (change.status === 'blocked') roles.push('blocked');
        if (change.urgent) roles.push('urgent');
        if (roles.length === 0) roles.push('other');
        return roles.join(' ');
    }
    
    // Utility Methods
    getInitials(name) {
        return name.split(' ').map(n => n[0]).join('').toUpperCase();
    }
    
    getAvatarColor(name) {
        const colors = [
            'linear-gradient(135deg, #007bff, #0056b3)',
            'linear-gradient(135deg, #dc3545, #c82333)',
            'linear-gradient(135deg, #28a745, #1e7e34)',
            'linear-gradient(135deg, #6f42c1, #5a2d91)',
            'linear-gradient(135deg, #fd7e14, #e55a00)',
            'linear-gradient(135deg, #17a2b8, #138496)'
        ];
        const hash = name.split('').reduce((a, b) => a + b.charCodeAt(0), 0);
        return colors[hash % colors.length];
    }
    
    estimateUserImpact(change) {
        // Mock user impact calculation - replace with real logic
        const impacts = {
            'PaymentGateway': '200K Users',
            'UserService': '150K Users',
            'PaymentAPI': '100K Users',
            'FrontendApp': '300K Users',
            'Database': '500K Users',
            'CoreAPI': '200K Users'
        };
        
        for (const service of change.services) {
            if (impacts[service]) return impacts[service];
        }
        return null;
    }
    
    // Filter Management
    setupFilters() {
        document.querySelectorAll('.role-filter').forEach(button => {
            button.addEventListener('click', this.handleFilterChange);
        });
    }
    
    handleFilterChange(event) {
        const button = event.target;
        const role = button.dataset.role;
        
        // Update active state
        document.querySelectorAll('.role-filter').forEach(btn => {
            btn.classList.remove('active');
            btn.setAttribute('aria-pressed', 'false');
        });
        button.classList.add('active');
        button.setAttribute('aria-pressed', 'true');
        
        this.currentFilter = role;
        this.applyCurrentFilter();
    }
    
    applyCurrentFilter() {
        const cards = document.querySelectorAll('.release-card');
        
        cards.forEach(card => {
            const cardRoles = card.dataset.role ? card.dataset.role.split(' ') : [];
            const shouldShow = this.currentFilter === 'all' || cardRoles.includes(this.currentFilter);
            card.closest('.col-xl-3').style.display = shouldShow ? 'block' : 'none';
        });
    }
    
    // Modal Management
    setupModalHandlers() {
        // Keyboard navigation
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.closeModal();
            }
        });
        
        // Click outside to close
        const modal = document.getElementById('workflowModal');
        if (modal) {
            modal.addEventListener('click', (e) => {
                if (e.target === modal) this.closeModal();
            });
        }
    }
    
    showWorkflow(changeId) {
        const change = this.changes.get(changeId);
        if (!change) return;
        
        document.getElementById('workflowTicket').textContent = changeId;
        document.getElementById('workflowModal').classList.add('show');
        
        // Update workflow diagram and details based on change data
        this.updateWorkflowModal(change);
        
        // Focus management for accessibility
        document.getElementById('workflowModal').focus();
    }
    
    updateWorkflowModal(change) {
        // This would update the SVG workflow diagram and step details
        // Implementation depends on your specific workflow visualization needs
        console.log('Updating workflow modal for:', change);
    }
    
    closeModal() {
        const modal = document.getElementById('workflowModal');
        if (modal) {
            modal.classList.remove('show');
        }
    }
    
    // Tooltip Management
    initializeTooltips() {
        document.querySelectorAll('[title]').forEach(element => {
            this.initializeTooltip(element);
        });
    }
    
    initializeCardTooltips(cardElement) {
        cardElement.querySelectorAll('[title]').forEach(element => {
            this.initializeTooltip(element);
        });
    }
    
    initializeTooltip(element) {
        // Clean up existing tooltip
        if (this.tooltips.has(element)) {
            this.tooltips.get(element).dispose();
        }
        
        // Create new tooltip
        const tooltip = new bootstrap.Tooltip(element);
        this.tooltips.set(element, tooltip);
    }
    
    // Status Updates
    updateConnectionStatus(status) {
        const statusElement = document.getElementById('connectionStatus');
        if (statusElement) {
            statusElement.textContent = status;
            statusElement.className = status === 'Connected' ? 'text-success' : 'text-danger';
        }
    }
    
    updateLastUpdated(timestamp = null) {
        const element = document.getElementById('lastUpdated');
        if (element) {
            if (timestamp) {
                const date = new Date(timestamp);
                element.textContent = date.toLocaleTimeString();
            } else {
                element.textContent = 'Live';
                setTimeout(() => {
                    if (element.textContent === 'Live') {
                        element.textContent = 'Just now';
                    }
                }, 1000);
            }
        }
    }
    
    // Cleanup
    destroy() {
        if (this.eventSource) {
            this.eventSource.close();
        }
        
        // Clean up tooltips
        this.tooltips.forEach(tooltip => tooltip.dispose());
        this.tooltips.clear();
        
        // Remove event listeners
        document.querySelectorAll('.role-filter').forEach(button => {
            button.removeEventListener('click', this.handleFilterChange);
        });
    }
}

// Global instance for easy access
let dashboard;

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    dashboard = new ReleaseDashboard();
    
    // Expose globally for button onclick handlers
    window.dashboard = dashboard;
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (dashboard) {
        dashboard.destroy();
    }
});