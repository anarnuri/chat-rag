document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('file-input');
    const messageForm = document.getElementById('message-form');
    const messageInput = document.getElementById('message-input');
    const chatMessages = document.getElementById('chat-messages');
    const submitBtn = messageForm.querySelector('button');
    
    let currentFileId = null;
    let isAssistantTyping = false;

    // Handle file upload
    fileInput.addEventListener('change', async (e) => {
        if (fileInput.files.length === 0) return;
        
        const file = fileInput.files[0];
        addMessage('assistant', `Processing ${file.name}...`);
        
        try {
            const formData = new FormData();
            formData.append('file', file);
            
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (!response.ok) {
                // Handle structured error responses
                const errorMsg = result.detail?.message || 
                                result.detail?.error || 
                                result.detail || 
                                'Upload failed';
                const errorStage = result.detail?.stage ? ` (failed at: ${result.detail.stage})` : '';
                throw new Error(`${errorMsg}${errorStage}`);
            }

            currentFileId = result.conversation_id;
            messageInput.disabled = false;
            submitBtn.disabled = false;
            
            // Clear chat and show success
            chatMessages.innerHTML = '';
            addMessage('assistant', `I've processed "${result.filename}". Ask me anything about it!`);
            
            // Log full response for debugging
            console.log('Upload successful:', result);
        } catch (error) {
            console.error('Upload error:', error);
            addMessage('assistant', `❌ ${formatErrorMessage(error)}`, true);
            
            // Show more details in console for debugging
            if (error.response) {
                error.response.json().then(errData => {
                    console.error('Full error details:', errData);
                });
            }
        } finally {
            fileInput.value = '';
        }
    });

    messageForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const question = messageInput.value.trim();
        
        if (!question) {
            addMessage('assistant', 'Please type a question', true);
            return;
        }
        if (!currentFileId) {
            addMessage('assistant', 'Please upload a file first', true);
            return;
        }
        
        console.log('SENDING conversation_id:', currentFileId); // Debug line
        
        // Add user message
        addMessage('user', question);
        messageInput.value = '';
        
        // Show typing indicator
        isAssistantTyping = true;
        const typingId = showTypingIndicator();
        
        try {
            // ===== PUT THE DEBUG CODE HERE =====
            console.log('Current conversation_id:', currentFileId);
            console.log('Question:', question);

            const response = await fetch('/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    conversation_id: String(currentFileId), // Ensure it's a string
                    question: question
                })
            }).catch(err => {
                console.error('Network error:', err);
                throw err;
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                console.error('Server error details:', errorData);
                throw new Error(errorData.detail || 'Request failed');
            }
            // ===== END OF DEBUG CODE =====

            const result = await response.json();
            
            // Remove typing indicator and show actual response
            removeTypingIndicator(typingId);
            addMessage('assistant', result.answer);
            
            console.log('API response:', result);
        } catch (error) {
            removeTypingIndicator(typingId);
            console.error('Ask error:', error);
            addMessage('assistant', `❌ ${formatErrorMessage(error)}`, true);
        } finally {
            isAssistantTyping = false;
        }
    });

    // Helper functions
    function addMessage(sender, text, isError = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        if (isError) {
            messageDiv.classList.add('error-message');
            messageDiv.innerHTML = `<div class="error-content">${text}</div>`;
        } else {
            messageDiv.textContent = text;
        }
        
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function formatErrorMessage(error) {
        // Handle different error formats
        if (error.message.includes('Failed to fetch')) {
            return 'Network error - please check your connection';
        }
        
        if (error.message.includes('Unexpected token')) {
            return 'Invalid server response';
        }
        
        return error.message;
    }

    function showTypingIndicator() {
        const id = 'typing-' + Date.now();
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message assistant-message';
        typingDiv.id = id;
        typingDiv.innerHTML = `
            <div class="typing-indicator">
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
                <span class="typing-text">Thinking...</span>
            </div>
        `;
        chatMessages.appendChild(typingDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return id;
    }

    function removeTypingIndicator(id) {
        const element = document.getElementById(id);
        if (element) element.remove();
    }
});