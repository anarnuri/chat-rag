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
            
            if (response.ok) {
                currentFileId = result.file_id;
                messageInput.disabled = false;
                submitBtn.disabled = false;
                
                // Clear chat and show success
                chatMessages.innerHTML = '';
                addMessage('assistant', `I've processed "${result.filename}". Ask me anything about it!`);
            } else {
                throw new Error(result.detail || 'Upload failed');
            }
        } catch (error) {
            addMessage('assistant', `Error: ${error.message}`, true);
        } finally {
            fileInput.value = '';
        }
    });

    // Handle message submission
    messageForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const question = messageInput.value.trim();
        
        if (!question || !currentFileId) return;
        
        // Add user message
        addMessage('user', question);
        messageInput.value = '';
        
        // Show typing indicator
        isAssistantTyping = true;
        const typingId = showTypingIndicator();
        
        try {
            const response = await fetch('/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    file_id: currentFileId,
                    question: question
                })
            });
            
            const result = await response.json();
            
            if (response.ok) {
                // Remove typing indicator and show actual response
                removeTypingIndicator(typingId);
                addMessage('assistant', result.answer);
            } else {
                throw new Error(result.detail || 'Failed to get answer');
            }
        } catch (error) {
            removeTypingIndicator(typingId);
            addMessage('assistant', `Error: ${error.message}`, true);
        } finally {
            isAssistantTyping = false;
        }
    });

    // Helper functions
    function addMessage(sender, text, isError = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        if (isError) messageDiv.style.color = 'red';
        
        const content = document.createElement('div');
        content.textContent = text;
        messageDiv.appendChild(content);
        
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
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