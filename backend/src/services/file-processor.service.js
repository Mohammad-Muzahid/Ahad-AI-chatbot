const pdf = require('pdf-parse');
const mammoth = require('mammoth');
const fs = require('fs-extra');
const path = require('path');
const { v4: uuidv4 } = require('uuid');

class FileProcessor {
    constructor() {
        this.supportedTypes = {
            // Images
            'image/jpeg': 'image',
            'image/png': 'image',
            'image/gif': 'image',
            'image/webp': 'image',
            
            // Documents
            'application/pdf': 'document',
            'text/plain': 'document',
            'text/markdown': 'document',
            'application/json': 'document',
            'text/csv': 'document',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'document',
            'application/msword': 'document'
        };
        
        this.uploadsDir = path.join(__dirname, '../../uploads');
        fs.ensureDirSync(this.uploadsDir);
    }

    async processFiles(files) {
        const results = [];
        
        for (const file of files) {
            try {
                console.log(`📄 Processing file: ${file.originalname} (${file.mimetype})`);
                
                const result = await this.processFile(file);
                results.push(result);
                
            } catch (error) {
                console.error(`❌ Error processing file ${file.originalname}:`, error.message);
                results.push({
                    filename: file.originalname,
                    type: this.getFileType(file.mimetype),
                    size: file.size,
                    content: null,
                    error: error.message,
                    success: false
                });
            }
        }
        
        return results;
    }

    async processFile(file) {
        const fileId = uuidv4();
        const fileType = this.getFileType(file.mimetype);
        const fileExtension = path.extname(file.originalname).toLowerCase();
        
        // Save file to disk (optional, for persistence)
        const savedFilePath = path.join(this.uploadsDir, `${fileId}${fileExtension}`);
        await fs.writeFile(savedFilePath, file.buffer);
        
        // Extract content based on file type
        let content = null;
        let preview = null;
        
        if (fileType === 'image') {
            // For images, we'll create a description using OCR (basic for now)
            content = await this.processImage(file);
            preview = `data:${file.mimetype};base64,${file.buffer.toString('base64')}`;
            
        } else if (fileType === 'document') {
            // For documents, extract text content
            content = await this.extractDocumentContent(file, fileExtension);
        }
        
        return {
            id: fileId,
            filename: file.originalname,
            type: fileType,
            subtype: file.mimetype,
            size: file.size,
            content: content,
            preview: preview,
            path: savedFilePath,
            success: true,
            timestamp: new Date().toISOString()
        };
    }

    async processImage(file) {
    try {
        // Create a more descriptive analysis of the image
        const imageInfo = {
            filename: file.originalname,
            size: this.formatFileSize(file.size),
            type: file.mimetype,
            dimensions: 'Unknown' // Would need image processing library to get dimensions
        };
        
        // Create a prompt for the image
        // In a production system, you would use:
        // 1. Tesseract.js for OCR
        // 2. TensorFlow.js for object detection
        // 3. CLIP for image understanding
        
        const imageDescription = `
[IMAGE FILE ANALYSIS]
Filename: ${imageInfo.filename}
File Type: ${imageInfo.type}
File Size: ${imageInfo.size}

This is an image file that has been uploaded. To analyze this image properly:

1. VISUAL DESCRIPTION NEEDED: The image appears to contain visual content that requires analysis.
2. POTENTIAL CONTENT: It could contain text, objects, people, diagrams, or other visual elements.
3. ANALYSIS REQUESTED: When the user asks about this image, provide a detailed visual description including:
   - Objects and elements visible
   - Colors and composition
   - Any readable text
   - Context and possible meaning
   - Overall impression and observations

For specific analysis, please ask the user to describe what they see or ask specific questions about the image.
`;

        return imageDescription;
        
    } catch (error) {
        console.error('Image processing error:', error);
        return `[Image: ${file.originalname}] Image file uploaded. To analyze it, please describe what you see or ask specific questions about the image content.`;
    }
}

formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

    async extractDocumentContent(file, extension) {
        try {
            const buffer = file.buffer;
            
            switch (extension) {
                case '.pdf':
                    return await this.extractPDF(buffer);
                    
                case '.docx':
                case '.doc':
                    return await this.extractDOCX(buffer);
                    
                case '.txt':
                case '.md':
                    return buffer.toString('utf-8');
                    
                case '.json':
                    try {
                        const jsonContent = JSON.parse(buffer.toString('utf-8'));
                        return `JSON Data: ${JSON.stringify(jsonContent, null, 2)}`;
                    } catch (e) {
                        return buffer.toString('utf-8');
                    }
                    
                case '.csv':
                    return await this.extractCSV(buffer);
                    
                default:
                    // Try to extract as plain text
                    try {
                        return buffer.toString('utf-8');
                    } catch (e) {
                        return `[Document: ${file.originalname}] Unable to extract text content.`;
                    }
            }
        } catch (error) {
            console.error(`Document extraction error for ${file.originalname}:`, error);
            return `[Document: ${file.originalname}] Error extracting content: ${error.message}`;
        }
    }

    async extractPDF(buffer) {
        try {
            const data = await pdf(buffer);
            return data.text;
        } catch (error) {
            console.error('PDF extraction error:', error);
            return '[PDF] Unable to extract text content. The PDF may be scanned or encrypted.';
        }
    }

    async extractDOCX(buffer) {
        try {
            const result = await mammoth.extractRawText({ buffer: buffer });
            return result.value;
        } catch (error) {
            console.error('DOCX extraction error:', error);
            return '[DOCX] Unable to extract text content.';
        }
    }

    async extractCSV(buffer) {
        try {
            const text = buffer.toString('utf-8');
            const lines = text.split('\n').filter(line => line.trim());
            
            if (lines.length === 0) return '[CSV] Empty file';
            
            // Format CSV for better readability
            const formatted = lines.map(line => {
                const columns = line.split(',');
                return columns.join(' | ');
            }).join('\n');
            
            return `CSV Data (${lines.length} rows):\n${formatted}`;
        } catch (error) {
            console.error('CSV extraction error:', error);
            return buffer.toString('utf-8');
        }
    }

    getFileType(mimeType) {
        return this.supportedTypes[mimeType] || 'unknown';
    }

    async cleanupOldFiles(days = 7) {
        try {
            const files = await fs.readdir(this.uploadsDir);
            const now = Date.now();
            const cutoff = now - (days * 24 * 60 * 60 * 1000);
            
            for (const file of files) {
                const filePath = path.join(this.uploadsDir, file);
                const stats = await fs.stat(filePath);
                
                if (stats.mtimeMs < cutoff) {
                    await fs.unlink(filePath);
                    console.log(`🗑️  Cleaned up old file: ${file}`);
                }
            }
        } catch (error) {
            console.error('Cleanup error:', error);
        }
    }
}

module.exports = { FileProcessor };