// Global variables
let isProcessing = false;

// DOM elements
const questionInput = document.getElementById('question-input');
const sendButton = document.getElementById('send-button');
const chatMessages = document.getElementById('chat-messages');
const typingIndicator = document.getElementById('typing-indicator');
const loadingOverlay = document.getElementById('loading-overlay');
const statusIndicator = document.getElementById('status-indicator');

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    console.log('App initializing...');
    initializeApp();
    setupEventListeners();
    checkServerHealth();
});

function initializeApp() {
    console.log('Setting up app...');
    
    // Focus on input
    if (questionInput) {
        questionInput.focus();
    }
    
    // Setup suggestion buttons
    setupSuggestionButtons();
    
    // Load topics dynamically
    loadTopics();
}

function setupEventListeners() {
    console.log('Setting up event listeners...');
    
    // Send button click
    if (sendButton) {
        sendButton.addEventListener('click', function(e) {
            e.preventDefault();
            console.log('Send button clicked');
            handleSendMessage();
        });
    }
    
    // Enter key press
    if (questionInput) {
        questionInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                console.log('Enter key pressed');
                handleSendMessage();
            }
        });
        
        // Input change to enable/disable send button
        questionInput.addEventListener('input', function() {
            const hasText = this.value.trim().length > 0;
            if (sendButton) {
                sendButton.disabled = !hasText || isProcessing;
            }
        });
    }
    
    // Topic card clicks
    document.addEventListener('click', function(e) {
        const topicCard = e.target.closest('.topic-card');
        if (topicCard) {
            console.log('Topic card clicked:', topicCard.dataset.topic);
            highlightTopic(topicCard);
        }
    });
}

function setupSuggestionButtons() {
    const suggestionButtons = document.querySelectorAll('.suggestion-btn');
    console.log('Found suggestion buttons:', suggestionButtons.length);
    
    suggestionButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            e.preventDefault();
            const question = this.dataset.question;
            console.log('Suggestion clicked:', question);
            if (questionInput) {
                questionInput.value = question;
                questionInput.focus();
                handleSendMessage();
            }
        });
    });
}

async function handleSendMessage() {
    if (!questionInput || !sendButton) {
        console.error('Input elements not found');
        return;
    }
    
    const question = questionInput.value.trim();
    console.log('Handling send message:', question);
    
    if (!question || isProcessing) {
        console.log('No question or already processing');
        return;
    }
    
    try {
        // Clear input and disable send button
        questionInput.value = '';
        sendButton.disabled = true;
        isProcessing = true;
        
        // Add user message to chat
        addMessageToChat('user', question);
        
        // Show typing indicator
        showTypingIndicator();
        
        console.log('Sending request to backend...');
        
        // Send request to backend
        const response = await fetch('/api/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query: question })
        });
        
        console.log('Response status:', response.status);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('Response data:', data);
        
        if (data.success) {
            // Add assistant response to chat
            addMessageToChat('assistant', data.answer, {
                topic: data.detected_topic,
                subtopic: data.detected_subtopic,
                sourceCount: data.source_count,
                sources: data.sources
            });
        } else {
            // Add error message
            addMessageToChat('assistant', `Sorry, I encountered an error: ${data.error}`, { isError: true });
        }
        
    } catch (error) {
        console.error('Error sending message:', error);
        addMessageToChat('assistant', 'Sorry, I encountered a connection error. Please check the console and try again.', { isError: true });
    } finally {
        // Hide typing indicator and re-enable input
        hideTypingIndicator();
        isProcessing = false;
        if (sendButton) {
            sendButton.disabled = false;
        }
        if (questionInput) {
            questionInput.focus();
        }
    }
}

function addMessageToChat(sender, text, metadata = {}) {
    if (!chatMessages) {
        console.error('Chat messages container not found');
        return;
    }
    
    console.log('Adding message:', sender, text.substring(0, 50) + '...');
    
    // Remove welcome message if this is the first real message
    const welcomeMessage = document.querySelector('.welcome-message');
    if (welcomeMessage && sender === 'user') {
        welcomeMessage.remove();
    }
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;
    
    let avatarIcon = sender === 'user' ? 'fa-user' : 'fa-robot';
    let messageHTML = `
        <div class="message-avatar">
            <i class="fas ${avatarIcon}"></i>
        </div>
        <div class="message-content">
            <div class="message-text">${formatMessageText(text)}</div>
    `;
    
    // Add metadata for assistant messages
    if (sender === 'assistant' && !metadata.isError) {
        messageHTML += `<div class="message-meta">`;
        
        if (metadata.topic) {
            messageHTML += `<span class="topic-badge">${metadata.topic}`;
            if (metadata.subtopic) {
                messageHTML += ` → ${metadata.subtopic}`;
            }
            messageHTML += `</span>`;
        }
        
        if (metadata.sourceCount) {
            messageHTML += `<div class="sources-info">
                <i class="fas fa-books"></i> Based on ${metadata.sourceCount} knowledge sources
            </div>`;
        }
        
        messageHTML += `</div>`;
    }
    
    messageHTML += `</div>`;
    messageDiv.innerHTML = messageHTML;
    
    chatMessages.appendChild(messageDiv);
    
    // Scroll to bottom with smooth animation
    setTimeout(() => {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }, 100);
    
    // Add fade-in animation
    messageDiv.style.opacity = '0';
    messageDiv.style.transform = 'translateY(20px)';
    messageDiv.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
    
    setTimeout(() => {
        messageDiv.style.opacity = '1';
        messageDiv.style.transform = 'translateY(0)';
    }, 50);
}

function formatMessageText(text) {
    // Convert newlines to HTML breaks and format basic markdown
    return text
        .replace(/\n/g, '<br>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/`(.*?)`/g, '<code>$1</code>');
}

function showTypingIndicator() {
    if (typingIndicator) {
        typingIndicator.style.display = 'flex';
    }
}

function hideTypingIndicator() {
    if (typingIndicator) {
        typingIndicator.style.display = 'none';
    }
}

function highlightTopic(topicCard) {
    // Remove previous highlights
    document.querySelectorAll('.topic-card').forEach(card => {
        card.classList.remove('active');
    });
    
    // Add highlight to clicked topic
    topicCard.classList.add('active');
    
    // Add some visual feedback
    topicCard.style.transform = 'scale(1.02)';
    setTimeout(() => {
        topicCard.style.transform = '';
    }, 200);
}

async function loadTopics() {
    try {
        const response = await fetch('/api/topics');
        const data = await response.json();
        console.log('Topics loaded:', data);
        
        if (data.topics) {
            updateTopicsDisplay(data.topics);
        }
    } catch (error) {
        console.error('Error loading topics:', error);
    }
}

function updateTopicsDisplay(topics) {
    const topicsList = document.getElementById('topics-list');
    if (!topicsList) return;
    
    const topicIcons = {
        'DBMS': 'fa-database',
        'OOPs': 'fa-code', 
        'OS': 'fa-cogs'
    };
    
    const topicNames = {
        'DBMS': 'Database Management Systems',
        'OOPs': 'Object-Oriented Programming',
        'OS': 'Operating Systems'
    };
    
    // Clear existing topics
    topicsList.innerHTML = '';
    
    topics.forEach(topic => {
        const topicCard = document.createElement('div');
        topicCard.className = 'topic-card';
        topicCard.dataset.topic = topic.name;
        
        const subtopicsHTML = topic.subtopics.slice(0, 4).map(subtopic => 
            `<span class="subtopic">${subtopic}</span>`
        ).join('');
        
        topicCard.innerHTML = `
            <div class="topic-header">
                <i class="fas ${topicIcons[topic.name] || 'fa-tag'}"></i>
                <span>${topicNames[topic.name] || topic.name}</span>
            </div>
            <div class="subtopics">
                ${subtopicsHTML}
            </div>
        `;
        
        topicsList.appendChild(topicCard);
    });
}

async function checkServerHealth() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        console.log('Health check:', data);
        
        if (data.status === 'healthy' && data.rag_initialized) {
            updateStatusIndicator('ready', 'Ready');
        } else {
            updateStatusIndicator('warning', 'Initializing...');
        }
    } catch (error) {
        updateStatusIndicator('error', 'Connection Error');
        console.error('Health check failed:', error);
    }
}

function updateStatusIndicator(status, text) {
    if (!statusIndicator) return;
    
    const indicator = statusIndicator.querySelector('i');
    
    // Remove all status classes
    if (indicator) {
        indicator.classList.remove('status-ready', 'status-warning', 'status-error');
        indicator.classList.add(`status-${status}`);
    }
    
    // Update text
    statusIndicator.innerHTML = `<i class="fas fa-circle status-${status}"></i> ${text}`;
}

// Utility functions
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

console.log('App.js loaded successfully');
