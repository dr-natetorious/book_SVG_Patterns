class ReleaseCard extends HTMLElement {
    constructor() {
      super();
      this.attachShadow({ mode: 'open' });
    }
  
    connectedCallback() {
      const change = JSON.parse(this.getAttribute('data-change'));
      this.renderCard(change);
    }
  
    renderCard(change) {
      const cardHtml = `
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
      this.shadowRoot.innerHTML = cardHtml;
    }
  
    getCardClasses(change) {
      const classes = ['release-card'];
      if (change.status === 'running') classes.push('running');
      if (change.status === 'my-change') classes.push('my-change');
      return classes.join(' ');
    }
  
    getDataRoles(change) {
      const roles = [];
      if (change.is_my_change) roles.push('my');
      return roles.join(' ');
    }
  
    generateCardHeader(change) {
      return `
        <div class="card-header">
          <div class="d-flex align-items-center justify-content-between">
            <div class="d-flex align-items-center">
              <div class="risk-indicator ${this.getRiskClass(change.risk)}" 
                   role="img" 
                   aria-label="${this.getRiskLabel(change.risk)}"
                   title="${this.getRiskTitle(change.risk)}">${this.getRiskIcon(change.risk)}</div>
              <div>
                <div class="fw-bold text-dark" style="font-size: 0.9rem;" id="change-${change.id}-title">${change.id}</div>
                <div style="font-size: 0.8rem; color: #6c757d;">${change.services.join(', ')}</div>
              </div>
            </div>
            <div class="maintenance-status ${this.getMaintenanceClass(change.status)}" 
                 aria-label="${this.getMaintenanceLabel(change.status)}">
              <i class="bi ${this.getMaintenanceIcon(change.status)}" aria-hidden="true"></i> 
              <span>${change.maintenance_window}</span>
            </div>
          </div>
        </div>
      `;
    }
  
    generateCardBody(change) {
      return `
        <div class="card-body">
          ${this.generateCurrentStatus(change)}
          ${this.generateImpactIndicators(change)}
          <div class="scroll-hint" id="scrollHint${change.id}">
            <i class="bi bi-chevron-double-down" aria-hidden="true"></i> Scroll
          </div>
        </div>
      `;
    }
  
    generateCurrentStatus(change) {
      return `
        <div class="current-status ${this.getStatusClass(change.status)}">
          <div class="status-title">
            <i class="bi ${this.getStatusIcon(change.status)}" aria-hidden="true"></i>
            <span>${this.getStatusTitle(change)}</span>
            ${this.getStatusSpinner(change.status)}
          </div>
          <div class="status-description">
            ${this.getStatusDescription(change)}
          </div>
          ${this.getProgressBar(change)}
          <div class="next-action">
            <i class="bi ${this.getNextActionIcon(change.status)}" aria-hidden="true"></i> ${this.getNextActionText(change.status)}
          </div>
        </div>
      `;
    }
  
    generateImpactIndicators(change) {
      return `
        <div class="impact-indicators">
          <span class="impact-badge impact-services">
            <i class="bi bi-server" aria-hidden="true"></i> ${change.services.length} Services
          </span>
          <span class="impact-badge impact-users">
            <i class="bi bi-people" aria-hidden="true"></i> ${change.user_impact} Users
          </span>
        </div>
      `;
    }
  
    generateCardFooter(change) {
      return `
        <div class="card-footer">
          <div class="d-flex justify-content-between align-items-center">
            <div class="assignee-info">
              <div class="assignee-avatar" title="${change.assignee}">${this.getAssigneeInitials(change.assignee)}</div>
              <span>${change.assignee}</span>
              <span class="time-estimate">ETA: ${change.eta}</span>
            </div>
            <button class="btn btn-outline-primary btn-sm" 
                    onclick="showWorkflow('${change.id}')"
                    aria-label="View workflow diagram for ${change.id}">
              <i class="bi bi-diagram-3" aria-hidden="true"></i> Workflow
            </button>
          </div>
        </div>
      `;
    }
  
    getRiskClass(risk) {
      const riskClasses = {
        low: 'risk-low',
        medium: 'risk-medium',
        high: 'risk-high',
        critical: 'risk-critical'
      };
      return riskClasses[risk];
    }
  
    getRiskLabel(risk) {
      const riskLabels = {
        low: 'Low risk change',
        medium: 'Medium risk change with striped pattern indicator',
        high: 'High risk change with diagonal stripes',
        critical: 'Critical risk change with cross-hatch pattern'
      };
      return riskLabels[risk];
    }
  
    getRiskTitle(risk) {
      const riskTitles = {
        low: 'Low Risk - Auto-approved',
        medium: 'Medium Risk - Manager approval required',
        high: 'High Risk - Director approval required',
        critical: 'Critical Risk - Emergency only'
      };
      return riskTitles[risk];
    }
  
    getRiskIcon(risk) {
      const riskIcons = {
        low: 'L',
        medium: 'M',
        high: 'H',
        critical: 'C'
      };
      return riskIcons[risk];
    }
  
    getMaintenanceClass(status) {
      const maintenanceClasses = {
        running: 'maintenance-active'
      };
      return maintenanceClasses[status];
    }
    
    getMaintenanceIcon(status) {
      const maintenanceIcons = {
        running: 'bi-lightning-charge'
      };
      return maintenanceIcons[status];
    }
  
    getMaintenanceLabel(status) {
      const maintenanceLabels = {
        running: 'Currently in maintenance window'
      };
      return maintenanceLabels[status];
    }
  
    getStatusClass(status) {
      const statusClasses = {
        running: 'running'
      };
      return statusClasses[status];
    }
  
    getStatusIcon(status) {
      const statusIcons = {
        running: 'bi-gear-fill text-primary'
      };
      return statusIcons[status];
    }
  
    getStatusTitle(change) {
      const statusTitles = {
        running: `Deploying - Step ${change.current_step} of ${change.total_steps}`
      };
      return statusTitles[change.status];
    }
  
    getStatusSpinner(status) {
      const statusSpinners = {
        running: `
          <div class="spinner-border spinner-border-sm text-primary" role="status" aria-label="Deployment in progress">
            <span class="sr-only">Loading...</span>
          </div>
        `
      };
      return statusSpinners[status];
    }
  
    getStatusDescription(change) {
      const statusDescriptions = {
        running: `Upgrading backend services - ETA ${change.eta} minutes`
      };
      return statusDescriptions[change.status];
    }
  
    getProgressBar(change) {
      return `
        <div class="progress mb-2" style="height: 6px;" role="progressbar" aria-valuenow="${change.progress}" aria-valuemin="0" aria-valuemax="100">
          <div class="progress-bar bg-primary progress-bar-striped progress-bar-animated" style="width: ${change.progress}%"></div>
        </div>
      `;
    }
  
    getNextActionIcon(status) {
      const nextActionIcons = {
        running: 'bi-clock'
      };
      return nextActionIcons[status];
    }
  
    getNextActionText(status) {
      const nextActionTexts = {
        running: 'Monitoring deployment progress'
      };
      return nextActionTexts[status];
    }
  
    getAssigneeInitials(assignee) {
      return assignee.split(' ').map(n => n[0]).join('').toUpperCase();
    }
  }
  
  customElements.define('release-card', ReleaseCard);