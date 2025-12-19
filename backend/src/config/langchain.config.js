module.exports = {
    // LLM Configuration
    llm: {
        provider: process.env.LLM_PROVIDER || 'openai',
        model: process.env.LLM_MODEL || 'gpt-3.5-turbo',
        temperature: parseFloat(process.env.LLM_TEMPERATURE || '0.7'),
        maxTokens: parseInt(process.env.LLM_MAX_TOKENS || '1000'),
        timeout: parseInt(process.env.LLM_TIMEOUT || '30000')
    },
    
    // Embeddings Configuration
    embeddings: {
        provider: process.env.EMBEDDINGS_PROVIDER || 'openai',
        model: process.env.EMBEDDINGS_MODEL || 'text-embedding-ada-002',
        dimensions: parseInt(process.env.EMBEDDINGS_DIMENSIONS || '1536')
    },
    
    // Vector Store Configuration
    vectorStore: {
        type: process.env.VECTOR_STORE_TYPE || 'chroma',
        collectionName: process.env.VECTOR_STORE_COLLECTION || 'ahad_knowledge',
        persistDirectory: process.env.VECTOR_STORE_DIR || './vector_store'
    },
    
    // RAG Configuration
    rag: {
        chunkSize: parseInt(process.env.RAG_CHUNK_SIZE || '1000'),
        chunkOverlap: parseInt(process.env.RAG_CHUNK_OVERLAP || '200'),
        similarityTopK: parseInt(process.env.RAG_SIMILARITY_K || '4'),
        similarityThreshold: parseFloat(process.env.RAG_SIMILARITY_THRESHOLD || '0.7')
    },
    
    // Memory Configuration
    memory: {
        maxHistory: parseInt(process.env.MEMORY_MAX_HISTORY || '50'),
        summaryInterval: parseInt(process.env.MEMORY_SUMMARY_INTERVAL || '10')
    },
    
    // Voice Configuration
    voice: {
        sttProvider: process.env.STT_PROVIDER || 'browser',
        ttsProvider: process.env.TTS_PROVIDER || 'browser',
        defaultLanguage: process.env.DEFAULT_LANGUAGE || 'en',
        sampleRate: parseInt(process.env.VOICE_SAMPLE_RATE || '16000')
    }
};