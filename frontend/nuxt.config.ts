export default defineNuxtConfig({
  compatibilityDate: '2024-11-01',
  devtools: { enabled: false },
  modules: ['@nuxtjs/tailwindcss'],
  runtimeConfig: {
    public: {
      apiBase: process.env.NUXT_PUBLIC_API_BASE || 'http://localhost:8000'
    }
  },
  app: {
    head: {
      title: 'SemanticDoc — Local PyTorch & RAG Studio',
      meta: [
        { name: 'description', content: 'Local PyTorch + Hugging Face + FAISS + Ollama RAG Document Assistant' }
      ]
    }
  }
})

