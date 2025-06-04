// Mock data structure for SSE integration
const mockChanges = {
    'PROJ-1234': {
        id: 'PROJ-1234',
        title: 'Fix payment processing bug',
        services: ['UserService', 'PaymentAPI'],
        risk: 'medium',
        status: 'deploying',
        currentStep: 3,
        totalSteps: 5,
        progress: 60,
        assignee: 'Sarah Chen',
        maintenanceWindow: 'MW-2024-001',
        eta: '12 minutes',
        isMyChange: true,
        steps: [
            { name: 'Approvals', status: 'complete', completedAt: '2h ago', details: 'PM ✓ Sarah Chen, Dev ✓ John Smith, QA ✓ Maria Rodriguez' },
            { name: 'Window', status: 'complete', completedAt: 'Active now', details: 'Dec 16, 2024 02:00-06:00 GMT • Remaining: 3h 15m' },
            { name: 'Merge PRs', status: 'running', completedAt: 'In Progress', details: '1 of 2 PRs merged • Current: BB-4522 (Backend API changes) • ETA: 3 minutes' },
            { name: 'Deploy', status: 'notstarted', completedAt: 'Waiting', details: '5 deployment actions • Services: UserService, PaymentAPI • Estimated Duration: 15 minutes' },
            { name: 'Verify', status: 'notstarted', completedAt: 'Pending', details: 'Health endpoints, integration tests, monitoring alerts • Duration: 5 minutes' }
        ]
    },
    'PROJ-1235': {
        id: 'PROJ-1235',
        title: 'Database schema migration',
        services: ['Database', 'CoreAPI'],
        risk: 'high',
        status: 'blocked',
        assignee: 'John Smith',
        blockers: ['Missing QA approval - Maria Rodriguez', 'No maintenance window scheduled'],
        isMyChange: false
    },
    'PROJ-1236': {
        id: 'PROJ-1236',
        title: 'Frontend UI updates',
        services: ['FrontendApp'],
        risk: 'low',
        status: 'ready',
        assignee: 'Alex Wong',
        maintenanceWindow: 'MW-2024-002',
        eta: 'Tonight',
        isMyChange: false
    },
    'PROJ-1237': {
        id: 'PROJ-1237',
        title: 'Payment gateway rollback',
        services: ['PaymentGateway'],
        risk: 'critical',
        status: 'rollback',
        currentStep: 2,
        totalSteps: 3,
        progress: 67,
        assignee: 'Sarah Chen',
        eta: '5 minutes',
        isMyChange: true,
        urgent: true
    }
};

// Role filter functionality
document.querySelectorAll('.role-filter').forEach(button => {
    button.addEventListener('click', function() {
        // Update active state
        document.querySelectorAll('.role-filter').forEach(btn => {
            btn.classList.remove('active');
            btn.setAttribute('aria-pressed', 'false');
        });
        this.classList.add('active');
        this.setAttribute('aria-pressed', 'true');

        // Filter cards
        const role = this.dataset.role;
        const cards = document.querySelectorAll('.release-card');
        
        cards.forEach(card => {
            const cardRoles = card.dataset.role ? card.dataset.role.split(' ') : [];
            const shouldShow = role === 'all' || cardRoles.includes(role);
            card.closest('.col-xl-3').style.display = shouldShow ? 'block' : 'none';
        });
    });
});

// Check if card bodies need scroll indicators
function updateScrollIndicators() {
    document.querySelectorAll('.card-body').forEach((body, index) => {
        const hasScroll = body.scrollHeight > body.clientHeight;
        const scrollHint = body.querySelector('.scroll-hint');
        
        if (hasScroll) {
            body.classList.add('has-scroll');
            if (scrollHint && body.scrollTop === 0) {
                scrollHint.classList.add('show');
            }
        } else {
            body.classList.remove('has-scroll');
            if (scrollHint) {
                scrollHint.classList.remove('show');
            }
        }
    });
}

// Handle scroll events to hide/show scroll hints
function setupScrollListeners() {
    document.querySelectorAll('.card-body').forEach(body => {
        const scrollHint = body.querySelector('.scroll-hint');
        
        body.addEventListener('scroll', function() {
            if (scrollHint) {
                // Hide hint when user starts scrolling
                if (this.scrollTop > 10) {
                    scrollHint.classList.remove('show');
                } else {
                    // Show hint again when at top and has more content
                    const hasScroll = this.scrollHeight > this.clientHeight;
                    if (hasScroll) {
                        scrollHint.classList.add('show');
                    }
                }
            }
        });

        // Touch event handling for mobile swipe gestures
        let startY = 0;
        let startTime = 0;

        body.addEventListener('touchstart', function(e) {
            startY = e.touches[0].clientY;
            startTime = Date.now();
        }, { passive: true });

        body.addEventListener('touchmove', function(e) {
            // Allow native scrolling
        }, { passive: true });

        body.addEventListener('touchend', function(e) {
            const endY = e.changedTouches[0].clientY;
            const endTime = Date.now();
            const deltaY = startY - endY;
            const deltaTime = endTime - startTime;

            // Detect swipe gestures for quick scrolling
            if (Math.abs(deltaY) > 50 && deltaTime < 300) {
                const scrollAmount = deltaY > 0 ? 100 : -100;
                this.scrollBy({
                    top: scrollAmount,
                    behavior: 'smooth'
                });
            }
        }, { passive: true });
    });
}

// Modal functions
function showWorkflow(changeId) {
    const change = mockChanges[changeId];
    if (!change) return;

    document.getElementById('workflowTicket').textContent = changeId;
    document.getElementById('workflowModal').classList.add('show');
    
    // Update SVG workflow diagram based on change status
    updateWorkflowDiagram(change);
    
    // Update step details
    updateStepDetails(change);

    // Focus management for accessibility
    document.getElementById('workflowModal').focus();
}

function updateWorkflowDiagram(change) {
    const svg = document.querySelector('.workflow-diagram');
    const steps = svg.querySelectorAll('g[id^="step"]');
    
    if (change.steps) {
        change.steps.forEach((step, index) => {
            const stepElement = steps[index];
            const circle = stepElement.querySelector('circle');
            const text = stepElement.querySelector('text');
            
            // Reset animations
            circle.style.animation = '';
            
            switch(step.status) {
                case 'complete':
                    circle.setAttribute('fill', '#198754');
                    text.textContent = '✓';
                    break;
                case 'running':
                    circle.setAttribute('fill', '#6f42c1');
                    circle.style.animation = 'pulse 1.5s ease-in-out infinite';
                    text.textContent = '⚙';
                    break;
                case 'pending':
                    circle.setAttribute('fill', '#0d6efd');
                    text.textContent = '○';
                    break;
                case 'failed':
                    circle.setAttribute('fill', '#dc3545');
                    text.textContent = '✗';
                    break;
                default:
                    circle.setAttribute('fill', '#6c757d');
                    text.textContent = '○';
            }
        });
    }
}

function updateStepDetails(change) {
    const detailsContainer = document.querySelector('.step-details');
    if (!change.steps) return;

    // This would be populated with actual step details from the change data
    // For now, the static HTML handles the display
}

function closeModal(event) {
    if (event && event.target !== event.currentTarget) return;
    document.getElementById('workflowModal').classList.remove('show');
}

// SSE connection simulation
function simulateSSE() {
    // This would connect to your actual SSE endpoint
    // For now, simulate updates
    setInterval(() => {
        const randomChange = Object.keys(mockChanges)[Math.floor(Math.random() * Object.keys(mockChanges).length)];
        const change = mockChanges[randomChange];
        
        if (change.status === 'deploying' || change.status === 'rollback') {
            // Simulate progress updates
            change.progress = Math.min(100, change.progress + Math.random() * 10);
            if (change.progress >= 100) {
                change.status = 'complete';
                change.currentStep = change.totalSteps;
            }
        }
        
        // Update UI would happen here based on SSE data
        updateLastUpdated();
    }, 5000);
}

function updateLastUpdated() {
    document.getElementById('lastUpdated').textContent = 'Live';
    setTimeout(() => {
        document.getElementById('lastUpdated').textContent = '5s ago';
    }, 1000);
}

// Initialize everything
document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[title]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Setup scroll functionality
    updateScrollIndicators();
    setupScrollListeners();

    // Update scroll indicators on window resize
    window.addEventListener('resize', updateScrollIndicators);

    // Start SSE simulation
    simulateSSE();
});

// Keyboard navigation for accessibility
document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
        closeModal();
    }
});

// High contrast mode detection
if (window.matchMedia && window.matchMedia('(prefers-contrast: high)').matches) {
    document.body.classList.add('high-contrast');
}