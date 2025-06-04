class WorkflowModal extends HTMLElement {
    constructor() {
      super();
      this.attachShadow({ mode: 'open' });
      this.modal = null;
      this.ticket = null;
    }
  
    connectedCallback() {
      this.shadowRoot.innerHTML = `
        <div class="workflow-modal" id="workflowModal" role="dialog" aria-labelledby="workflowModalTitle" aria-modal="true">
          <div class="workflow-content" onclick="event.stopPropagation()">
            <div class="workflow-header">
                <div>
                    <h4 class="mb-1 fw-bold" id="workflowModalTitle">Workflow: <span id="workflowTicket">PROJ-1234</span></h4>
                    <p class="mb-0 text-muted">Release Pipeline Progress</p>
                </div>
                <button class="btn btn-outline-secondary btn-sm" onclick="closeModal()" aria-label="Close workflow diagram">
                    <i class="bi bi-x-lg" aria-hidden="true"></i>
                </button>
            </div>
            <div class="workflow-body">
                <svg class="workflow-diagram" viewBox="0 0 800 200" role="img" aria-labelledby="workflowDiagramTitle">
                    <title id="workflowDiagramTitle">Release workflow diagram showing 5 steps from approval to deployment</title>
                    
                    <!-- Workflow Steps -->
                    <!-- Step 1: Approvals -->
                    <g id="step1">
                        <circle cx="80" cy="100" r="25" fill="#198754" stroke="#fff" stroke-width="3"/>
                        <text x="80" y="95" text-anchor="middle" fill="white" font-size="16" font-weight="bold">✓</text>
                        <text x="80" y="140" text-anchor="middle" font-size="12" font-weight="600">Approvals</text>
                        <line x1="105" y1="100" x2="135" y2="100" stroke="#198754" stroke-width="3"/>
                    </g>
                    
                    <!-- Step 2: Maintenance Window -->
                    <g id="step2">
                        <circle cx="160" cy="100" r="25" fill="#198754" stroke="#fff" stroke-width="3"/>
                        <text x="160" y="95" text-anchor="middle" fill="white" font-size="16" font-weight="bold">✓</text>
                        <text x="160" y="140" text-anchor="middle" font-size="12" font-weight="600">Window</text>
                        <line x1="185" y1="100" x2="215" y2="100" stroke="#198754" stroke-width="3"/>
                    </g>
                    
                    <!-- Step 3: Merge PRs -->
                    <g id="step3">
                        <circle cx="240" cy="100" r="25" fill="#6f42c1" stroke="#fff" stroke-width="3" stroke-dasharray="5,5">
                            <animate attributeName="stroke-dashoffset" values="0;10" dur="1s" repeatCount="indefinite"/>
                        </circle>
                        <text x="240" y="95" text-anchor="middle" fill="white" font-size="16" font-weight="bold">⚙</text>
                        <text x="240" y="140" text-anchor="middle" font-size="12" font-weight="600">Merge PRs</text>
                        <line x1="265" y1="100" x2="295" y2="100" stroke="#6c757d" stroke-width="2" stroke-dasharray="3,3"/>
                    </g>
                    
                    <!-- Step 4: Deploy -->
                    <g id="step4">
                        <circle cx="320" cy="100" r="25" fill="#6c757d" stroke="#fff" stroke-width="3"/>
                        <text x="320" y="95" text-anchor="middle" fill="white" font-size="16" font-weight="bold">○</text>
                        <text x="320" y="140" text-anchor="middle" font-size="12" font-weight="600">Deploy</text>
                        <line x1="345" y1="100" x2="375" y2="100" stroke="#6c757d" stroke-width="2" stroke-dasharray="3,3"/>
                    </g>
                    
                    <!-- Step 5: Verify -->
                    <g id="step5">
                        <circle cx="400" cy="100" r="25" fill="#6c757d" stroke="#fff" stroke-width="3"/>
                        <text x="400" y="95" text-anchor="middle" fill="white" font-size="16" font-weight="bold">○</text>
                        <text x="400" y="140" text-anchor="middle" font-size="12" font-weight="600">Verify</text>
                    </g>
                    
                    <!-- Legend -->
                    <g id="legend" transform="translate(500, 50)">
                        <text x="0" y="0" font-size="14" font-weight="bold">Status Legend:</text>
                        <circle cx="15" cy="20" r="8" fill="#198754"/>
                        <text x="30" y="25" font-size="12">Complete</text>
                        <circle cx="15" cy="40" r="8" fill="#6f42c1" stroke-dasharray="2,2">
                            <animate attributeName="stroke-dashoffset" values="0;4" dur="0.5s" repeatCount="indefinite"/>
                        </circle>
                        <text x="30" y="45" font-size="12">Running</text>
                        <circle cx="15" cy="60" r="8" fill="#0d6efd"/>
                        <text x="30" y="65" font-size="12">Pending</text>
                        <circle cx="15" cy="80" r="8" fill="#dc3545"/>
                        <text x="30" y="85" font-size="12">Failed</text>
                        <circle cx="15" cy="100" r="8" fill="#6c757d"/>
                        <text x="30" y="105" font-size="12">Not Started</text>
                    </g>
                </svg>

                <div class="step-details">
                    <h5 class="fw-bold mb-3">Step Details</h5>
                    
                    <div class="step-detail-item">
                        <div class="step-detail-header">
                            <div class="step-status-icon status-complete-icon" role="img" aria-label="Complete">✓</div>
                            <h6 class="mb-0 fw-bold">1. Change Approval Process</h6>
                            <small class="text-success ms-auto">Completed 2h ago</small>
                        </div>
                        <p class="mb-2 text-muted">All required approvals obtained for deployment.</p>
                        <div class="small">
                            <strong>Details:</strong> PM ✓ Sarah Chen, Dev ✓ John Smith, QA ✓ Maria Rodriguez
                        </div>
                    </div>

                    <div class="step-detail-item">
                        <div class="step-detail-header">
                            <div class="step-status-icon status-complete-icon" role="img" aria-label="Complete">✓</div>
                            <h6 class="mb-0 fw-bold">2. Maintenance Window</h6>
                            <small class="text-success ms-auto">Active now</small>
                        </div>
                        <p class="mb-2 text-muted">Currently within scheduled maintenance window MW-2024-001.</p>
                        <div class="small">
                            <strong>Window:</strong> Dec 16, 2024 02:00-06:00 GMT • 
                            <strong>Remaining:</strong> 3h 15m
                        </div>
                    </div>

                    <div class="step-detail-item">
                        <div class="step-detail-header">
                            <div class="step-status-icon status-running-icon" role="img" aria-label="Currently running">⚙</div>
                            <h6 class="mb-0 fw-bold">3. Merge Pull Requests</h6>
                            <small class="text-primary ms-auto">In Progress</small>
                        </div>
                        <p class="mb-2 text-muted">Merging approved pull requests into production branches.</p>
                        <div class="small">
                            <strong>Progress:</strong> 1 of 2 PRs merged • 
                            <strong>Current:</strong> BB-4522 (Backend API changes) • 
                            <strong>ETA:</strong> 3 minutes
                        </div>
                    </div>

                    <div class="step-detail-item">
                        <div class="step-detail-header">
                            <div class="step-status-icon status-notstarted-icon" role="img" aria-label="Not started">○</div>
                            <h6 class="mb-0 fw-bold">4. Deployment Instructions</h6>
                            <small class="text-muted ms-auto">Waiting</small>
                        </div>
                        <p class="mb-2 text-muted">Execute deployment steps across target environments.</p>
                        <div class="small">
                            <strong>Steps:</strong> 5 deployment actions • 
                            <strong>Services:</strong> UserService, PaymentAPI • 
                            <strong>Estimated Duration:</strong> 15 minutes
                        </div>
                    </div>

                    <div class="step-detail-item">
                        <div class="step-detail-header">
                            <div class="step-status-icon status-notstarted-icon" role="img" aria-label="Not started">○</div>
                            <h6 class="mb-0 fw-bold">5. Post-Deployment Verification</h6>
                            <small class="text-muted ms-auto">Pending</small>
                        </div>
                        <p class="mb-2 text-muted">Verify deployment success and system health.</p>
                        <div class="small">
                            <strong>Checks:</strong> Health endpoints, integration tests, monitoring alerts • 
                            <strong>Duration:</strong> 5 minutes
                        </div>
                    </div>
                </div>
            </div>
        </div>
        </div>
      `;
      this.modal = this.shadowRoot.getElementById('workflowModal');
      this.ticket = this.shadowRoot.getElementById('workflowTicket');
    }
  
    show(ticket) {
      this.ticket.textContent = ticket;
      this.modal.classList.add('show');
      this.modal.focus();
    }
  
    close() {
      this.modal.classList.remove('show');
    }
  
    updateWorkflow(change) {
      // Update the workflow diagram and step details based on the change data
      // This would involve updating the SVG workflow diagram and the step details HTML
      console.log('Updating workflow modal for:', change);
    }
  }
  
  customElements.define('workflow-modal', WorkflowModal);