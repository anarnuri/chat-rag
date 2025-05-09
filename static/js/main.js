document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('upload-form');
    const fileInput = document.getElementById('file-input');
    const fileList = document.getElementById('file-list');
    
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        if (fileInput.files.length === 0) {
            alert('Please select at least one file');
            return;
        }
        
        // Process each file individually
        for (let i = 0; i < fileInput.files.length; i++) {
            const file = fileInput.files[i];
            await uploadFile(file);
        }
        
        // Clear the file input after upload
        fileInput.value = '';
    });
    
    async function uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);
        
        // Create UI element for this file
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        fileItem.innerHTML = `
            <div>${file.name}</div>
            <div class="progress">
                <div class="progress-bar" id="progress-${file.name}"></div>
            </div>
            <div id="status-${file.name}">Uploading...</div>
        `;
        fileList.appendChild(fileItem);
        
        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData,
                // Add progress tracking
                headers: {
                    'X-File-Name': encodeURIComponent(file.name)
                }
            });
            
            const result = await response.json();
            document.getElementById(`status-${file.name}`).textContent = 'Uploaded successfully';
            document.getElementById(`progress-${file.name}`).style.width = '100%';
            
        } catch (error) {
            document.getElementById(`status-${file.name}`).textContent = 'Error: ' + error.message;
            document.getElementById(`progress-${file.name}`).style.backgroundColor = '#f44336';
        }
    }
});