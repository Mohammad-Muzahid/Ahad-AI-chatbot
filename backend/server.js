const express = require('express');
const cors = require('cors');
const multer = require('multer');
const path = require('path');
const fs = require('fs-extra');
require('dotenv').config();

// Import services
const { AhadAIService } = require('./src/services/ahad-ai.service');
const { FileProcessor } = require('./src/services/file-processor.service');

const app = express();
const PORT = process.env.PORT || 3000;

// Configure multer for file uploads
const storage = multer.memoryStorage();
const upload = multer({
    storage: storage,
    limits: {
        fileSize: 10 * 1024 * 1024, // 10MB limit
        files: 5 // Maximum 5 files per request
    },
    fileFilter: (req, file, cb) => {
        const allowedTypes = [
            'image/jpeg',
            'image/png',
            'image/gif',
            'image/webp',
            'application/pdf',
            'text/plain',
            'text/markdown',
            'application/json',
            'text/csv',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/msword'
        ];
        
        if (allowedTypes.includes(file.mimetype)) {
            cb(null, true);
        } else {
            cb(new Error('Invalid file type. Only images, PDFs, text files, and documents are allowed.'));
        }
    }
});

// Middleware
app.use(cors({
    origin: ['http://localhost:5500', 'http://127.0.0.1:5500', 'http://localhost:3000'],
    credentials: true
}));
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Initialize services
const ahadAI = new AhadAIService();
const fileProcessor = new FileProcessor();

// Create uploads directory if it doesn't exist
const uploadsDir = path.join(__dirname, 'uploads');
fs.ensureDirSync(uploadsDir);

// Health check
app.get('/api/health', (req, res) => {
    res.json({
        status: 'healthy',
        service: 'Ahad AI Backend',
        version: '2.1.0',
        features: ['RAG', 'LangChain', 'Voice', 'File Upload', 'Multilingual'],
        timestamp: new Date().toISOString()
    });
});

// Initialize AI Service
(async () => {
    try {
        await ahadAI.initialize();
        console.log('✅ Ahad AI Service initialized successfully');
    } catch (error) {
        console.error('❌ Failed to initialize AI service:', error);
    }
})();

// Store active sessions with file context
const activeSessions = new Map();

// Main chat endpoint with file support
app.post('/api/chat', upload.array('files', 5), async (req, res) => {
    try {
        const {
            message,
            language = 'en',
            sessionId = 'default_' + Date.now(),
            useRAG = true,
            analyzeSentiment = true
        } = req.body;

        const files = req.files || [];
        
        if (!message && files.length === 0) {
            return res.status(400).json({ 
                success: false, 
                error: 'Either message or files are required' 
            });
        }

        console.log(`💬 Processing chat: "${message || 'File upload'}" with ${files.length} file(s)`);
        console.log(`📂 Session ID: ${sessionId}`);

        // Process files if any
        let fileResults = [];
        if (files.length > 0) {
            fileResults = await fileProcessor.processFiles(files);
            
            // Store file context in session
            if (!activeSessions.has(sessionId)) {
                activeSessions.set(sessionId, {
                    files: [],
                    messages: [],
                    createdAt: new Date().toISOString()
                });
            }
            
            const session = activeSessions.get(sessionId);
            session.files = [...session.files, ...fileResults];
            
            console.log(`💾 Stored ${fileResults.length} file(s) in session ${sessionId}`);
            
            // Store processed content in knowledge base
            for (const file of fileResults) {
                if (file.content) {
                    await ahadAI.addToKnowledgeBase(file.content, 'file_upload', {
                        filename: file.filename,
                        type: file.type,
                        size: file.size,
                        sessionId: sessionId
                    });
                }
            }
        }

        // Get file context from session if available
        let sessionFiles = [];
        if (activeSessions.has(sessionId)) {
            sessionFiles = activeSessions.get(sessionId).files;
        }

        // Process the message with AI - PASS FILE CONTEXT
        let aiResponse;
        if (message) {
            aiResponse = await ahadAI.processWithRAG({
                message,
                language,
                useRAG,
                analyzeSentiment,
                sessionId,
                files: sessionFiles, // Pass all files in session
                newFiles: fileResults // Pass newly uploaded files
            });
        } else {
            // Generate response based only on files
            const fileSummary = fileResults.map(f => 
                `📄 ${f.filename} (${f.type}): ${f.content ? 'Content extracted' : 'Unable to extract content'}`
            ).join('\n');
            
            aiResponse = await ahadAI.processWithRAG({
                message: `I uploaded these files: ${fileSummary}. Analyze them and tell me about the content.`,
                language,
                useRAG: true,
                analyzeSentiment: false,
                sessionId,
                files: sessionFiles,
                newFiles: fileResults
            });
        }

        // Update session messages
        if (!activeSessions.has(sessionId)) {
            activeSessions.set(sessionId, {
                files: fileResults,
                messages: [],
                createdAt: new Date().toISOString()
            });
        }
        
        const session = activeSessions.get(sessionId);
        session.messages.push({
            role: 'user',
            content: message,
            files: fileResults.map(f => f.filename),
            timestamp: new Date().toISOString()
        });
        
        session.messages.push({
            role: 'assistant',
            content: aiResponse.text,
            timestamp: new Date().toISOString()
        });

        // Keep only last 20 messages
        if (session.messages.length > 20) {
            session.messages = session.messages.slice(-20);
        }

        // Combine file info with AI response
        const response = {
            success: true,
            ...aiResponse,
            sessionId: sessionId,
            files: fileResults.map(f => ({
                filename: f.filename,
                type: f.type,
                size: f.size,
                extracted: !!f.content,
                preview: f.preview,
                success: f.success
            })),
            sessionFileCount: sessionFiles.length,
            timestamp: new Date().toISOString()
        };

        res.json(response);

    } catch (error) {
        console.error('Chat error:', error);
        res.status(500).json({
            success: false,
            error: error.message || 'Internal server error',
            timestamp: new Date().toISOString()
        });
    }
});

// Get session files
app.get('/api/session/:sessionId/files', async (req, res) => {
    try {
        const { sessionId } = req.params;
        
        if (!activeSessions.has(sessionId)) {
            return res.json({
                success: true,
                sessionId,
                files: [],
                message: 'No active session found'
            });
        }
        
        const session = activeSessions.get(sessionId);
        
        res.json({
            success: true,
            sessionId,
            files: session.files.map(f => ({
                filename: f.filename,
                type: f.type,
                size: f.size,
                extracted: !!f.content
            })),
            count: session.files.length,
            timestamp: new Date().toISOString()
        });
        
    } catch (error) {
        console.error('Get session files error:', error);
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

// Clear session
app.delete('/api/session/:sessionId', async (req, res) => {
    try {
        const { sessionId } = req.params;
        
        if (activeSessions.has(sessionId)) {
            activeSessions.delete(sessionId);
        }
        
        res.json({
            success: true,
            message: 'Session cleared successfully',
            sessionId,
            timestamp: new Date().toISOString()
        });
        
    } catch (error) {
        console.error('Clear session error:', error);
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

// File upload endpoint (separate from chat)
app.post('/api/upload', upload.array('files', 5), async (req, res) => {
    try {
        const { sessionId = 'default_' + Date.now() } = req.body;
        const files = req.files;
        
        if (!files || files.length === 0) {
            return res.status(400).json({
                success: false,
                error: 'No files uploaded'
            });
        }

        console.log(`📤 Processing ${files.length} file(s) for upload to session: ${sessionId}`);

        const results = await fileProcessor.processFiles(files);
        
        // Store in session
        if (!activeSessions.has(sessionId)) {
            activeSessions.set(sessionId, {
                files: [],
                messages: [],
                createdAt: new Date().toISOString()
            });
        }
        
        const session = activeSessions.get(sessionId);
        session.files = [...session.files, ...results];
        
        // Add to knowledge base
        for (const file of results) {
            if (file.content) {
                await ahadAI.addToKnowledgeBase(file.content, 'file_upload', {
                    filename: file.filename,
                    type: file.type,
                    size: file.size,
                    sessionId: sessionId
                });
            }
        }

        res.json({
            success: true,
            message: `Successfully processed ${files.length} file(s)`,
            sessionId: sessionId,
            files: results.map(f => ({
                id: f.id,
                filename: f.filename,
                type: f.type,
                size: f.size,
                extracted: !!f.content,
                preview: f.preview,
                timestamp: new Date().toISOString()
            })),
            timestamp: new Date().toISOString()
        });

    } catch (error) {
        console.error('Upload error:', error);
        res.status(500).json({
            success: false,
            error: error.message || 'File processing failed',
            timestamp: new Date().toISOString()
        });
    }
});

// Other endpoints remain the same...
app.post('/api/rag/query', async (req, res) => {
    try {
        const { query, language = 'en', topK = 3 } = req.body;

        if (!query) {
            return res.status(400).json({
                success: false,
                error: 'Query is required'
            });
        }

        console.log(`🔍 RAG Query: "${query}"`);

        const results = await ahadAI.ragQuery(query, language, { topK });

        res.json({
            success: true,
            query,
            results,
            timestamp: new Date().toISOString()
        });

    } catch (error) {
        console.error('RAG query error:', error);
        res.status(500).json({
            success: false,
            error: error.message || 'RAG query failed'
        });
    }
});

// Knowledge base management
app.post('/api/knowledge/ingest', async (req, res) => {
    try {
        const { text, source = 'manual', metadata = {} } = req.body;

        if (!text) {
            return res.status(400).json({
                success: false,
                error: 'Text is required'
            });
        }

        const result = await ahadAI.addToKnowledgeBase(text, source, metadata);

        res.json({
            success: true,
            message: 'Knowledge added successfully',
            documentId: result.ids?.[0],
            count: result.count,
            timestamp: new Date().toISOString()
        });

    } catch (error) {
        console.error('Knowledge ingestion error:', error);
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

// Sentiment analysis
app.post('/api/sentiment', async (req, res) => {
    try {
        const { text, language = 'en' } = req.body;

        if (!text) {
            return res.status(400).json({
                success: false,
                error: 'Text is required'
            });
        }

        const sentiment = await ahadAI.analyzeSentiment(text, language);

        res.json({
            success: true,
            text,
            sentiment,
            language,
            timestamp: new Date().toISOString()
        });

    } catch (error) {
        console.error('Sentiment analysis error:', error);
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

// Service status
app.get('/api/status', async (req, res) => {
    try {
        const status = ahadAI.getStatus();
        
        res.json({
            success: true,
            ...status,
            activeSessions: activeSessions.size,
            timestamp: new Date().toISOString()
        });
    } catch (error) {
        console.error('Status check error:', error);
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

// Error handling middleware
app.use((err, req, res, next) => {
    console.error('Global error:', err);
    
    if (err instanceof multer.MulterError) {
        if (err.code === 'LIMIT_FILE_SIZE') {
            return res.status(400).json({
                success: false,
                error: 'File too large. Maximum size is 10MB.'
            });
        }
        if (err.code === 'LIMIT_FILE_COUNT') {
            return res.status(400).json({
                success: false,
                error: 'Too many files. Maximum is 5 files.'
            });
        }
    }
    
    res.status(500).json({
        success: false,
        error: 'Internal server error',
        message: err.message
    });
});

// Start server
app.listen(PORT, () => {
    console.log(`🚀 Ahad AI Backend Server running on port ${PORT}`);
    console.log(`📊 Health: http://localhost:${PORT}/api/health`);
    console.log(`💬 Chat API: http://localhost:${PORT}/api/chat`);
    console.log(`📤 Upload API: http://localhost:${PORT}/api/upload`);
    console.log(`📚 RAG System: Ready for file uploads and analysis`);
});