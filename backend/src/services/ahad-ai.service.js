const { Chroma } = require('@langchain/community/vectorstores/chroma');
const { OllamaEmbeddings } = require('@langchain/community/embeddings/ollama');
const { RecursiveCharacterTextSplitter } = require('langchain/text_splitter');
const { ChatOllama } = require('@langchain/community/chat_models/ollama');
const natural = require('natural');

class AhadAIService {
    constructor() {
        this.vectorStore = null;
        this.llm = null;
        this.embeddings = null;
        this.isReady = false;
        this.localKnowledge = [];
        this.conversationHistory = new Map();
        this.chromaInitialized = false;
        this.fileKnowledge = new Map();
        this.sessionContexts = new Map();
        
        // Language configuration
        this.languageConfig = {
            'en': {
                name: 'English',
                greeting: 'Hello! I\'m Ahad AI, your multilingual assistant. How can I help you today?',
                instructions: {
                    general: 'Provide helpful, accurate responses in English.',
                    image: 'Describe the image content in detail.',
                    document: 'Analyze the document and extract key information.',
                    file: 'Reference uploaded files when relevant.'
                }
            },
            'hi': {
                name: 'Hindi',
                greeting: 'नमस्ते! मैं आहद AI हूं, आपका बहुभाषी सहायक। आज मैं आपकी कैसे सहायता कर सकता हूं?',
                instructions: {
                    general: 'हिंदी में सहायक, सटीक प्रतिक्रियाएं प्रदान करें।',
                    image: 'छवि सामग्री का विस्तार से वर्णन करें।',
                    document: 'दस्तावेज़ का विश्लेषण करें और मुख्य जानकारी निकालें।',
                    file: 'प्रासंगिक होने पर अपलोड की गई फ़ाइलों का संदर्भ दें।'
                }
            },
            'ar': {
                name: 'Arabic',
                greeting: 'مرحبًا! أنا آحاد AI، مساعدك متعدد اللغات. كيف يمكنني مساعدتك اليوم؟',
                instructions: {
                    general: 'قدم ردودًا مفيدة ودقيقة باللغة العربية.',
                    image: 'صف محتوى الصورة بالتفصيل.',
                    document: 'حلل المستند واستخرج المعلومات الرئيسية.',
                    file: 'أشر إلى الملفات المرفوعة عندما تكون ذات صلة.'
                }
            },
            'te': {
                name: 'Telugu',
                greeting: 'హలో! నేను ఆహద్ AI, మీ బహుభాషా సహాయకుడు. నేను ఈరోజు మీకు ఎలా సహాయం చేయగలను?',
                instructions: {
                    general: 'తెలుగులో సహాయకరమైన, ఖచ్చితమైన ప్రతిస్పందనలను అందించండి.',
                    image: 'చిత్రం కంటెంట్‌ను వివరంగా వివరించండి.',
                    document: 'డాక్యుమెంట్‌ను విశ్లేషించి ప్రధాన సమాచారాన్ని సేకరించండి.',
                    file: 'సంబంధితమైనప్పుడు అప్‌లోడ్ చేసిన ఫైళ్లను సూచించండి.'
                }
            }
        };
    }

    async initialize() {
        try {
            console.log('🚀 Initializing Ahad AI Service with Multilingual Support...');
            
            // 1. Initialize Ollama LLM
            this.llm = new ChatOllama({
                baseUrl: process.env.OLLAMA_BASE_URL || "http://localhost:11434",
                model: process.env.OLLAMA_MODEL || "llama2",
                temperature: 0.7,
                numPredict: 2000
            });
            console.log('✅ Ollama LLM initialized for multilingual support');

            // 2. Initialize embeddings
            this.embeddings = new OllamaEmbeddings({
                baseUrl: process.env.OLLAMA_BASE_URL || "http://localhost:11434",
                model: process.env.OLLAMA_EMBEDDING_MODEL || "nomic-embed-text"
            });
            console.log('✅ Ollama embeddings initialized');

            // 3. Initialize ChromaDB (optional)
            await this.initializeChromaDB();

            // 4. Load local knowledge
            await this.initializeLocalStorage();
            
            this.isReady = true;
            console.log('🎉 Ahad AI Service initialized successfully with 4 languages (EN, HI, AR, TE)');
            
        } catch (error) {
            console.error('❌ Failed to initialize AI service:', error.message);
            await this.initializeFallbackLLM();
            this.isReady = true;
        }
    }

    async initializeChromaDB() {
        try {
            const chromaUrl = process.env.CHROMA_URL || "http://localhost:8000";
            console.log(`🔗 Testing ChromaDB connection at: ${chromaUrl}`);
            
            const response = await fetch(`${chromaUrl}/api/v2/heartbeat`);
            if (!response.ok) {
                console.log('⚠️ ChromaDB not responding, skipping vector store');
                return;
            }
            
            console.log('💓 ChromaDB heartbeat test passed');
            
            try {
                this.vectorStore = await Chroma.fromExistingCollection(
                    this.embeddings,
                    {
                        collectionName: "ahad_knowledge",
                        url: chromaUrl
                    }
                );
                console.log('📚 Connected to existing ChromaDB collection');
                this.chromaInitialized = true;
            } catch (collectionError) {
                console.log('📝 ChromaDB collection not found');
                this.vectorStore = null;
            }
            
        } catch (chromaError) {
            console.log('⚠️ ChromaDB initialization failed:', chromaError.message);
            this.vectorStore = null;
        }
    }

    async initializeFallbackLLM() {
        try {
            this.llm = new ChatOllama({
                baseUrl: process.env.OLLAMA_BASE_URL || "http://localhost:11434",
                model: process.env.OLLAMA_MODEL || "llama2",
                temperature: 0.7,
                numPredict: 500
            });
            console.log('🤖 Fallback Ollama LLM initialized');
        } catch (error) {
            console.error('❌ Failed to initialize fallback LLM:', error);
        }
    }

    async initializeLocalStorage() {
        console.log('💾 Initializing multilingual knowledge storage...');
        
        // Multilingual knowledge base
        this.localKnowledge = [
            {
                content: "Ahad AI is a multilingual assistant supporting English, Hindi, Arabic, and Telugu",
                metadata: { 
                    source: "system", 
                    type: "multilingual",
                    languages: ["en", "hi", "ar", "te"]
                }
            },
            {
                content: "The assistant can analyze uploaded files including images, PDFs, and documents in any supported language",
                metadata: { 
                    source: "system", 
                    type: "file_support",
                    languages: ["en", "hi", "ar", "te"]
                }
            },
            {
                content: "For image analysis, describe visual elements, colors, objects, text, and overall composition",
                metadata: { 
                    source: "system", 
                    type: "image_analysis",
                    languages: ["en", "hi", "ar", "te"]
                }
            },
            {
                content: "For document analysis, summarize content, extract key points, identify themes, and answer specific questions",
                metadata: { 
                    source: "system", 
                    type: "document_analysis",
                    languages: ["en", "hi", "ar", "te"]
                }
            },
            {
                content: "The assistant maintains conversation context and can remember uploaded files within a session",
                metadata: { 
                    source: "system", 
                    type: "memory",
                    languages: ["en", "hi", "ar", "te"]
                }
            }
        ];
        
        console.log(`📚 Local knowledge initialized with ${this.localKnowledge.length} multilingual documents`);
    }

    async processWithRAG(options) {
        const {
            message,
            language = 'en',
            sessionId = 'default',
            useRAG = true,
            analyzeSentiment = true,
            files = [],
            newFiles = []
        } = options;

        console.log(`💬 Processing in ${language.toUpperCase()}: "${message}"`);
        console.log(`📂 Session: ${sessionId}, Files: ${files.length}`);

        try {
            // Get conversation history
            const history = await this.getConversationHistory(sessionId);
            
            // Get or create session context
            if (!this.sessionContexts.has(sessionId)) {
                this.sessionContexts.set(sessionId, {
                    files: [],
                    fileContext: '',
                    lastUpdated: new Date().toISOString(),
                    language: language
                });
            }
            
            const sessionContext = this.sessionContexts.get(sessionId);
            
            // Update session language if changed
            sessionContext.language = language;
            
            // Update session context with new files
            if (newFiles && newFiles.length > 0) {
                sessionContext.files = [...sessionContext.files, ...newFiles];
                
                // Build file context string
                let fileContext = '';
                newFiles.forEach(file => {
                    if (file.content) {
                        fileContext += `\n\n=== File: ${file.filename} ===\n`;
                        fileContext += `Type: ${file.type}\n`;
                        fileContext += `Size: ${this.formatFileSize(file.size)}\n`;
                        fileContext += `Content: ${file.content}\n`;
                    }
                });
                
                if (fileContext) {
                    sessionContext.fileContext += fileContext;
                }
                
                sessionContext.lastUpdated = new Date().toISOString();
                console.log(`📝 Updated session context with ${newFiles.length} new file(s)`);
            }

            // Process with RAG
            const ragResult = await this.ragQuery(
                message, 
                language, 
                { 
                    history,
                    fileContext: sessionContext.fileContext,
                    sessionId,
                    files: sessionContext.files
                }
            );

            // Analyze sentiment (English only for Natural NLP)
            const sentiment = analyzeSentiment && language === 'en' 
                ? await this.analyzeSentiment(message, language) 
                : 'neutral';
            
            // Detect intent
            const intent = this.detectIntent(message, language);

            // Store conversation
            await this.storeConversation(sessionId, {
                role: 'user',
                content: message,
                language: language,
                files: files?.length || 0,
                timestamp: new Date().toISOString()
            });
            
            await this.storeConversation(sessionId, {
                role: 'assistant',
                content: ragResult.text,
                language: language,
                timestamp: new Date().toISOString()
            });

            return {
                text: ragResult.text,
                sentiment: sentiment,
                intent: intent,
                sources: ragResult.sources || ['ollama'],
                confidence: ragResult.confidence || 0.8,
                language: language,
                shouldSpeak: true,
                sessionId: sessionId,
                fileCount: sessionContext.files.length,
                timestamp: new Date().toISOString()
            };

        } catch (error) {
            console.error('Process with RAG error:', error);
            return this.fallbackResponse(message, language);
        }
    }

    async ragQuery(query, language = 'en', context = {}) {
        console.log(`🔍 ${language.toUpperCase()} Query: "${query}"`);
        console.log(`📂 Context files: ${context.files ? context.files.length : 0}`);
        
        if (!this.isReady || !this.llm) {
            return this.fallbackResponse(query, language);
        }

        try {
            // Retrieve relevant documents
            let relevantDocs = [];
            let sources = [];
            
            // 1. Search local knowledge
            const queryWords = query.toLowerCase().split(' ').filter(w => w.length > 2);
            relevantDocs = this.localKnowledge
                .filter(doc => {
                    const content = doc.content.toLowerCase();
                    return queryWords.length === 0 || queryWords.some(word => content.includes(word));
                })
                .slice(0, 3)
                .map(doc => ({ 
                    pageContent: doc.content,
                    metadata: doc.metadata 
                }));
            
            // 2. Search ChromaDB if available
            if (this.vectorStore && this.chromaInitialized) {
                try {
                    const chromaDocs = await this.vectorStore.similaritySearch(query, 2);
                    relevantDocs = [...relevantDocs, ...chromaDocs];
                    sources.push('chromadb');
                } catch (searchError) {
                    console.log('⚠️ ChromaDB search error:', searchError.message);
                }
            }
            
            // 3. Search session-specific file knowledge
            if (context.sessionId && this.sessionContexts.has(context.sessionId)) {
                const sessionContext = this.sessionContexts.get(context.sessionId);
                if (sessionContext.files.length > 0) {
                    const fileDocs = this.searchInSessionFiles(query, sessionContext.files);
                    relevantDocs = [...relevantDocs, ...fileDocs];
                    if (fileDocs.length > 0) {
                        sources.push('session_files');
                    }
                }
            }
            
            sources.push('local_knowledge');
            console.log(`📄 Found ${relevantDocs.length} relevant documents`);

            // Generate response with Ollama - ENHANCED MULTILINGUAL PROMPT
            const contextText = relevantDocs.length > 0 
                ? relevantDocs.map(d => d.pageContent).join('\n\n')
                : 'No specific context available. Use your general knowledge.';
            
            // Get language configuration
            const langConfig = this.languageConfig[language] || this.languageConfig['en'];
            
            // Build enhanced prompt with language-specific instructions
            let enhancedPrompt = `You are Ahad AI, an intelligent multilingual assistant.

IMPORTANT: You MUST respond EXCLUSIVELY in ${langConfig.name} language (${language.toUpperCase()})!

AVAILABLE LANGUAGES: English (en), Hindi (hi), Arabic (ar), Telugu (te)
CURRENT RESPONSE LANGUAGE: ${langConfig.name} (${language.toUpperCase()})

CONTEXT INFORMATION:
${contextText}`;

            // Add file context if available
            if (context.fileContext && context.fileContext.trim()) {
                enhancedPrompt += `\n\nUPLOADED FILES INFORMATION:\n${context.fileContext}\n`;
            }

            if (context.files && context.files.length > 0) {
                enhancedPrompt += `\nCURRENT SESSION HAS ${context.files.length} UPLOADED FILE(S).`;
            }

            // Add conversation history
            if (context.history && context.history.trim() !== 'No previous conversation') {
                enhancedPrompt += `\n\nPREVIOUS CONVERSATION:\n${context.history}`;
            }

            // Add user query
            enhancedPrompt += `\n\nUSER QUERY: ${query}`;

            // Add language-specific instructions based on query type
            enhancedPrompt += `\n\nLANGUAGE-SPECIFIC INSTRUCTIONS:\n`;
            enhancedPrompt += `1. Respond ONLY in ${langConfig.name} (${language.toUpperCase()})\n`;
            enhancedPrompt += `2. ${langConfig.instructions.general}\n`;
            
            if (query.toLowerCase().includes('image') || query.toLowerCase().includes('picture') || query.toLowerCase().includes('photo')) {
                enhancedPrompt += `3. ${langConfig.instructions.image}\n`;
                enhancedPrompt += `   - Describe visual elements, colors, objects, text\n`;
                enhancedPrompt += `   - Mention composition and overall impression\n`;
                enhancedPrompt += `   - If text is visible, read and interpret it\n`;
            }
            
            if (query.toLowerCase().includes('document') || query.toLowerCase().includes('file') || query.toLowerCase().includes('pdf')) {
                enhancedPrompt += `3. ${langConfig.instructions.document}\n`;
                enhancedPrompt += `   - Summarize key points\n`;
                enhancedPrompt += `   - Extract important information\n`;
                enhancedPrompt += `   - Answer specific questions about content\n`;
            }
            
            if (context.files && context.files.length > 0) {
                enhancedPrompt += `3. ${langConfig.instructions.file}\n`;
            }
            
            // General guidelines
            enhancedPrompt += `\nGENERAL GUIDELINES:\n`;
            enhancedPrompt += `- Be helpful, friendly, and informative\n`;
            enhancedPrompt += `- If context is relevant, use it\n`;
            enhancedPrompt += `- If no context, use general knowledge\n`;
            enhancedPrompt += `- Maintain conversation flow\n`;
            enhancedPrompt += `- Keep responses concise but complete\n`;
            
            enhancedPrompt += `\nRESPONSE IN ${language.toUpperCase()}:\n`;

            console.log(`🤖 Generating ${language.toUpperCase()} response with enhanced context...`);
            const response = await this.llm.invoke(enhancedPrompt);
            
            return {
                text: response.content,
                sources: sources,
                confidence: 0.9,
                language: language,
                timestamp: new Date().toISOString()
            };
            
        } catch (error) {
            console.error(`❌ ${language.toUpperCase()} query error:`, error.message);
            return this.getBasicResponse(query, language);
        }
    }

    searchInSessionFiles(query, sessionFiles) {
        const results = [];
        const queryWords = query.toLowerCase().split(' ').filter(w => w.length > 2);
        
        if (queryWords.length === 0) return results;
        
        sessionFiles.forEach(file => {
            if (file.content) {
                const content = file.content.toLowerCase();
                const matches = queryWords.filter(word => content.includes(word)).length;
                
                if (matches > 0) {
                    results.push({
                        pageContent: `From uploaded file "${file.filename}":\n${file.content.substring(0, 1000)}...`,
                        metadata: {
                            source: 'session_file',
                            filename: file.filename,
                            type: file.type,
                            matchScore: matches / queryWords.length
                        }
                    });
                }
            }
        });
        
        return results.slice(0, 2);
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    async addToKnowledgeBase(text, source = 'user', metadata = {}) {
        try {
            // Add to local knowledge
            this.localKnowledge.push({
                content: text,
                metadata: {
                    source: source,
                    type: metadata.type || 'general',
                    timestamp: new Date().toISOString(),
                    languages: metadata.languages || ['en'], // Track supported languages
                    ...metadata
                }
            });
            
            console.log(`📝 Added to knowledge base from ${source}: ${text.substring(0, 100)}...`);
            
            // If ChromaDB is available, add there too
            if (this.vectorStore && this.chromaInitialized) {
                try {
                    const textSplitter = new RecursiveCharacterTextSplitter({
                        chunkSize: 1000,
                        chunkOverlap: 200
                    });
                    
                    const docs = await textSplitter.createDocuments([text], [metadata]);
                    await this.vectorStore.addDocuments(docs);
                    console.log('✅ Added to ChromaDB');
                } catch (chromaError) {
                    console.log('⚠️ Failed to add to ChromaDB:', chromaError.message);
                }
            }
            
            return {
                success: true,
                count: this.localKnowledge.length,
                ids: [this.localKnowledge.length - 1]
            };
            
        } catch (error) {
            console.error('❌ Failed to add to knowledge base:', error);
            return {
                success: false,
                error: error.message
            };
        }
    }

    // Conversation history methods
    async getConversationHistory(sessionId, limit = 10) {
        try {
            if (!this.conversationHistory.has(sessionId)) {
                return 'No previous conversation in this session.';
            }
            
            const history = this.conversationHistory.get(sessionId);
            const recentHistory = history.slice(-limit);
            
            return recentHistory.map(msg => 
                `${msg.role}: ${msg.content}${msg.files ? ` (uploaded ${msg.files} file(s))` : ''}`
            ).join('\n');
        } catch (error) {
            return 'No previous conversation.';
        }
    }

    async storeConversation(sessionId, message) {
        try {
            if (!this.conversationHistory.has(sessionId)) {
                this.conversationHistory.set(sessionId, []);
            }
            
            const history = this.conversationHistory.get(sessionId);
            history.push(message);
            
            // Limit to 50 messages
            if (history.length > 50) {
                this.conversationHistory.set(sessionId, history.slice(-50));
            }
        } catch (error) {
            console.error('Error storing conversation:', error);
        }
    }

    async analyzeSentiment(text, language = 'en') {
        try {
            // Only English sentiment analysis for now (Natural NLP is English-centric)
            if (language !== 'en') {
                return 'neutral';
            }
            
            const analyzer = new natural.SentimentAnalyzer('English', natural.PorterStemmer, 'afinn');
            const tokenizer = new natural.WordTokenizer();
            const tokens = tokenizer.tokenize(text);
            const score = analyzer.getSentiment(tokens);
            
            if (score > 0.2) return 'positive';
            if (score < -0.2) return 'negative';
            return 'neutral';
        } catch (error) {
            return 'neutral';
        }
    }

    detectIntent(text, language = 'en') {
        const textLower = text.toLowerCase();
        
        // Language-agnostic intent detection
        if (textLower.includes('upload') || textLower.includes('file') || textLower.includes('attach')) 
            return 'file_upload';
        if (textLower.includes('image') || textLower.includes('picture') || textLower.includes('photo')) 
            return 'image_analysis';
        if (textLower.includes('document') || textLower.includes('pdf') || textLower.includes('word')) 
            return 'document_analysis';
        if (textLower.includes('search') || textLower.includes('find')) 
            return 'search';
        if (textLower.includes('analyze') || textLower.includes('explain') || textLower.includes('describe')) 
            return 'analysis';
        if (textLower.includes('calculate') || textLower.includes('math')) 
            return 'calculate';
        if (textLower.includes('translate') || textLower.includes('language')) 
            return 'translate';
        if (textLower.includes('hello') || textLower.includes('hi') || 
            textLower.includes('नमस्ते') || textLower.includes('مرحبا') || textLower.includes('హలో')) 
            return 'greeting';
        if (textLower.includes('help')) 
            return 'help';
        
        return 'general';
    }

    async getBasicResponse(query, language) {
        if (this.llm) {
            try {
                const langConfig = this.languageConfig[language] || this.languageConfig['en'];
                
                const prompt = `User asked in ${langConfig.name}: "${query}"
                
Respond in ${langConfig.name} language only as Ahad AI, a helpful multilingual assistant.
Be friendly, informative, and helpful.
If the user is asking about uploaded files but you don't have file context, ask them to upload the file first.

Response in ${langConfig.name}:`;
                
                const response = await this.llm.invoke(prompt);
                return {
                    text: response.content,
                    sources: ['ollama_llm'],
                    confidence: 0.8,
                    language: language,
                    timestamp: new Date().toISOString()
                };
            } catch (error) {
                console.error('Ollama LLM error:', error);
            }
        }
        
        return this.fallbackResponse(query, language);
    }

    fallbackResponse(query, language) {
        const responses = {
            en: `I understand you're asking: "${query}". As Ahad AI, I can help you analyze uploaded files and answer questions about them. If you've uploaded files, please make sure they were successfully processed.`,
            hi: `मैं समझता हूं आप पूछ रहे हैं: "${query}"। आहद AI के रूप में, मैं अपलोड किए गए फाइलों का विश्लेषण करने और उनके बारे में प्रश्नों का उत्तर देने में आपकी सहायता कर सकता हूं। यदि आपने फाइलें अपलोड की हैं, तो कृपया सुनिश्चित करें कि वे सफलतापूर्वक प्रसंस्कृत हुई हैं।`,
            ar: `أفهم أنك تسأل: "${query}". كآحاد AI، يمكنني مساعدتك في تحليل الملفات المرفوعة والإجابة على أسئلتك عنها. إذا قمت بتحميل ملفات، يرجى التأكد من معالجتها بنجاح.`,
            te: `మీరు అడుగుతున్నారని నేను అర్థం చేసుకున్నాను: "${query}". ఆహద్ AI గా, నేను అప్‌లోడ్ చేసిన ఫైళ్లను విశ్లేషించడంలో మరియు వాటి గురించి ప్రశ్నలకు సమాధానం ఇవ్వడంలో మీకు సహాయపడగలను. మీరు ఫైళ్లను అప్‌లోడ్ చేసి ఉంటే, అవి విజయవంతంగా ప్రాసెస్ చేయబడ్డాయని నిర్ధారించుకోండి.`
        };
        
        return {
            text: responses[language] || responses.en,
            sources: ['fallback'],
            confidence: 0.6,
            language: language,
            timestamp: new Date().toISOString()
        };
    }

    getLanguageName(languageCode) {
        return this.languageConfig[languageCode]?.name || 'English';
    }

    getStatus() {
        return {
            isReady: this.isReady,
            hasLLM: !!this.llm,
            hasVectorStore: !!this.vectorStore && this.chromaInitialized,
            ollamaModel: process.env.OLLAMA_MODEL || "llama2",
            supportedLanguages: Object.keys(this.languageConfig),
            localKnowledgeCount: this.localKnowledge.length,
            activeSessions: this.sessionContexts.size,
            conversationSessions: this.conversationHistory.size
        };
    }

    // Clear session data
    async clearSession(sessionId) {
        try {
            if (this.sessionContexts.has(sessionId)) {
                this.sessionContexts.delete(sessionId);
            }
            
            if (this.conversationHistory.has(sessionId)) {
                this.conversationHistory.delete(sessionId);
            }
            
            return {
                success: true,
                message: `Session ${sessionId} cleared successfully`,
                timestamp: new Date().toISOString()
            };
        } catch (error) {
            console.error('Error clearing session:', error);
            return {
                success: false,
                error: error.message
            };
        }
    }

    // Get session information
    getSessionInfo(sessionId) {
        if (!this.sessionContexts.has(sessionId)) {
            return {
                exists: false,
                message: 'Session not found'
            };
        }
        
        const session = this.sessionContexts.get(sessionId);
        const history = this.conversationHistory.get(sessionId) || [];
        
        return {
            exists: true,
            sessionId: sessionId,
            language: session.language || 'en',
            fileCount: session.files?.length || 0,
            messageCount: history.length,
            lastUpdated: session.lastUpdated,
            timestamp: new Date().toISOString()
        };
    }
}

module.exports = { AhadAIService };