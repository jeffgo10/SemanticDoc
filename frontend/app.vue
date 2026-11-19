<template>
  <div class="min-h-screen bg-slate-950 text-slate-100 font-sans">
    <!-- Top Header / ML Engine Telemetry -->
    <header class="border-b border-slate-800 bg-slate-900/70 backdrop-blur sticky top-0 z-10">
      <div class="max-w-7xl mx-auto px-6 py-4 flex flex-wrap items-center justify-between gap-4">
        <div class="flex items-center gap-3">
          <div class="w-10 h-10 rounded-xl bg-indigo-600 flex items-center justify-center font-bold text-lg shadow-lg shadow-indigo-500/20">
            SD
          </div>
          <div>
            <h1 class="text-lg font-bold tracking-tight text-white">SemanticDoc Studio</h1>
            <p class="text-xs text-slate-400">Multi-PDF PyTorch + Hugging Face + FAISS + Ollama RAG</p>
          </div>
        </div>

        <!-- Live Health Telemetry Badges -->
        <div v-if="health" class="flex flex-wrap items-center gap-2 text-xs">
          <span class="px-2.5 py-1 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 flex items-center gap-1.5">
            <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
            API Online
          </span>
          <span class="px-2.5 py-1 rounded-full bg-slate-800 text-slate-300 border border-slate-700">
            PyTorch {{ health.pytorch_version }} ({{ health.compute_device.toUpperCase() }})
          </span>
          <span class="px-2.5 py-1 rounded-full bg-indigo-500/10 text-indigo-300 border border-indigo-500/20">
            HF: all-MiniLM-L6-v2
          </span>
          <span class="px-2.5 py-1 rounded-full bg-amber-500/10 text-amber-300 border border-amber-500/20">
            LLM: {{ health.llm_model }}
          </span>
        </div>
        <div v-else class="text-xs text-rose-400">
          Connecting to FastAPI backend at {{ apiBase }}...
        </div>
      </div>
    </header>

    <!-- Main Two-Column Layout -->
    <main class="max-w-7xl mx-auto px-6 py-8 grid grid-cols-1 lg:grid-cols-12 gap-8">
      <!-- Left Column: Multi-Document Library & PDF Upload -->
      <section class="lg:col-span-4 space-y-6">
        <!-- Indexed Documents Library Card -->
        <div class="p-5 rounded-2xl bg-slate-900 border border-slate-800 space-y-4">
          <div class="flex items-center justify-between">
            <div>
              <h2 class="text-sm font-semibold uppercase tracking-wider text-slate-400">
                Indexed PDFs in FAISS ({{ documents.length }})
              </h2>
              <p class="text-xs text-slate-500 mt-0.5">
                Total Vectors: <b class="text-indigo-400">{{ health?.total_chunks_indexed || 0 }}</b> (384D)
              </p>
            </div>
            <button
              @click="refreshAll"
              class="text-xs text-indigo-400 hover:text-indigo-300 transition"
            >
              Refresh
            </button>
          </div>

          <!-- Scope Selector Pill: All Docs vs Specific Doc -->
          <div class="space-y-2">
            <button
              @click="selectedDocFilter = ''"
              class="w-full flex items-center justify-between p-3 rounded-xl border text-left transition text-xs"
              :class="
                selectedDocFilter === ''
                  ? 'bg-indigo-600/15 border-indigo-500 text-white font-semibold'
                  : 'bg-slate-950 border-slate-800 text-slate-300 hover:bg-slate-800/60'
              "
            >
              <span class="flex items-center gap-2">
                <span>🌐</span>
                <span>Search Across All PDFs</span>
              </span>
              <span class="px-2 py-0.5 rounded-full bg-slate-800 text-slate-300 text-[11px]">
                {{ health?.total_chunks_indexed || 0 }} chunks
              </span>
            </button>

            <!-- Individual PDF Items -->
            <div
              v-for="doc in documents"
              :key="doc.filename"
              @click="selectedDocFilter = doc.filename"
              class="group flex items-center justify-between p-3 rounded-xl border cursor-pointer transition text-xs"
              :class="
                selectedDocFilter === doc.filename
                  ? 'bg-indigo-600/15 border-indigo-500 text-white'
                  : 'bg-slate-950 border-slate-800 text-slate-300 hover:bg-slate-800/60'
              "
            >
              <div class="min-w-0 pr-2">
                <div class="font-medium truncate flex items-center gap-1.5" :title="doc.filename">
                  <span>📘</span>
                  <span class="truncate">{{ doc.filename }}</span>
                </div>
                <div class="text-[11px] text-slate-500 mt-0.5">
                  {{ doc.pages }} pages · {{ doc.chunks }} vectors
                </div>
              </div>

              <button
                @click.stop="deletePdf(doc.filename)"
                title="Remove from FAISS"
                class="opacity-70 hover:opacity-100 px-2 py-1 rounded-lg hover:bg-rose-500/20 text-rose-400 transition"
              >
                🗑️
              </button>
            </div>

            <div
              v-if="documents.length === 0"
              class="text-center py-6 text-xs text-slate-500 border border-dashed border-slate-800 rounded-xl"
            >
              No PDFs indexed yet. Upload one below!
            </div>
          </div>
        </div>

        <!-- Upload PDF Card -->
        <div class="p-5 rounded-2xl bg-slate-900 border border-slate-800 space-y-4">
          <div>
            <h2 class="text-base font-semibold text-white">1. Add PDF to FAISS Store</h2>
            <p class="text-xs text-slate-400 mt-1">
              Tags chunks with <code class="text-indigo-300">doc_name</code>, embeds into 384D vectors via PyTorch, and appends to FAISS.
            </p>
          </div>

          <label
            class="flex flex-col items-center justify-center border-2 border-dashed border-slate-700 hover:border-indigo-500 rounded-xl p-6 cursor-pointer transition bg-slate-950/50"
          >
            <input type="file" accept=".pdf" class="hidden" @change="onFileSelected" />
            <div class="text-3xl mb-2">📄</div>
            <span class="text-sm font-medium text-slate-200 text-center">
              {{ selectedFile ? selectedFile.name : 'Click to select a PDF file' }}
            </span>
            <span class="text-xs text-slate-500 mt-1">Upload multiple PDFs to build your library</span>
          </label>

          <button
            :disabled="!selectedFile || uploading"
            @click="uploadPdf"
            class="w-full py-2.5 px-4 rounded-xl font-medium text-sm bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed transition text-white shadow-lg shadow-indigo-600/20"
          >
            {{ uploading ? 'Chunking & Embedding on PyTorch...' : 'Upload & Append to FAISS' }}
          </button>

          <!-- Upload Result Banner -->
          <div
            v-if="uploadResult"
            class="p-3.5 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-xs text-emerald-300 space-y-1"
          >
            <div class="font-semibold">✅ {{ uploadResult.message }}</div>
            <div>
              Pages: <b>{{ uploadResult.pages_processed }}</b> |
              Added: <b>{{ uploadResult.chunks_indexed }} chunks</b> |
              Time: <b>{{ uploadResult.processing_time_ms }} ms</b>
            </div>
          </div>

          <div
            v-if="uploadError"
            class="p-3.5 rounded-xl bg-rose-500/10 border border-rose-500/20 text-xs text-rose-300"
          >
            ⚠️ {{ uploadError }}
          </div>
        </div>

        <!-- Sample Preset Questions -->
        <div class="p-5 rounded-2xl bg-slate-900 border border-slate-800 space-y-3">
          <h2 class="text-xs font-semibold uppercase tracking-wider text-slate-400">
            Sample Questions
          </h2>
          <div class="space-y-2">
            <button
              v-for="(q, idx) in sampleQuestions"
              :key="idx"
              @click="askPreset(q)"
              class="w-full text-left text-xs p-3 rounded-xl bg-slate-950 hover:bg-slate-800 border border-slate-800 text-slate-300 transition"
            >
              “{{ q }}”
            </button>
          </div>
        </div>
      </section>

      <!-- Right Column: RAG Query & Retrieved Vector Chunks -->
      <section class="lg:col-span-8 space-y-6">
        <!-- Ask Question Box -->
        <div class="p-6 rounded-2xl bg-slate-900 border border-slate-800 space-y-4">
          <div class="flex flex-wrap items-center justify-between gap-3">
            <div>
              <h2 class="text-base font-semibold text-white">2. Ask the Multi-Doc RAG Agent</h2>
              <p class="text-xs text-slate-400 mt-0.5">
                Active Scope:
                <span class="text-indigo-400 font-semibold">
                  {{ selectedDocFilter ? `📘 ${selectedDocFilter} only` : '🌐 All Indexed PDFs' }}
                </span>
              </p>
            </div>

            <div class="flex items-center gap-3 text-xs text-slate-400">
              <div class="flex items-center gap-1.5">
                <span>Filter PDF:</span>
                <select
                  v-model="selectedDocFilter"
                  class="bg-slate-950 border border-slate-700 rounded-lg px-2.5 py-1 text-white"
                >
                  <option value="">All Documents</option>
                  <option v-for="doc in documents" :key="doc.filename" :value="doc.filename">
                    {{ doc.filename }}
                  </option>
                </select>
              </div>

              <div class="flex items-center gap-1.5">
                <span>Top-K:</span>
                <select
                  v-model.number="topK"
                  class="bg-slate-950 border border-slate-700 rounded-lg px-2.5 py-1 text-white"
                >
                  <option :value="3">3</option>
                  <option :value="5">5</option>
                  <option :value="8">8</option>
                </select>
              </div>
            </div>
          </div>

          <form @submit.prevent="submitQuestion" class="flex gap-3">
            <input
              v-model="question"
              type="text"
              placeholder="Ask anything across your indexed PDFs..."
              class="flex-1 bg-slate-950 border border-slate-700 focus:border-indigo-500 focus:outline-none rounded-xl px-4 py-3 text-sm text-white placeholder-slate-500"
            />
            <button
              type="submit"
              :disabled="!question.trim() || querying"
              class="px-6 py-3 rounded-xl font-medium text-sm bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed transition text-white whitespace-nowrap shadow-lg shadow-indigo-600/20"
            >
              {{ querying ? 'Thinking...' : 'Ask AI' }}
            </button>
          </form>

          <div
            v-if="queryError"
            class="p-3.5 rounded-xl bg-rose-500/10 border border-rose-500/20 text-xs text-rose-300"
          >
            ⚠️ {{ queryError }}
          </div>
        </div>

        <!-- Loading Skeleton -->
        <div
          v-if="querying"
          class="p-8 rounded-2xl bg-slate-900 border border-slate-800 flex flex-col items-center justify-center text-center space-y-3"
        >
          <div class="w-8 h-8 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin"></div>
          <div class="text-sm font-medium text-slate-200">
            Running PyTorch Embedding → FAISS Search → Ollama Generation...
          </div>
          <div class="text-xs text-slate-500">
            Searching {{ selectedDocFilter ? `'${selectedDocFilter}'` : 'all indexed PDFs' }}...
          </div>
        </div>

        <!-- Query Results -->
        <div v-if="queryResult && !querying" class="space-y-6">
          <!-- Grounded Answer Card -->
          <div class="p-6 rounded-2xl bg-slate-900 border border-indigo-500/30 space-y-4">
            <div class="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 pb-3">
              <div class="flex items-center gap-2">
                <span class="text-lg">🤖</span>
                <h3 class="text-sm font-bold text-white uppercase tracking-wider">
                  Generated Answer (Ollama)
                </h3>
                <span class="text-xs px-2 py-0.5 rounded-full bg-slate-800 text-slate-300">
                  Scope: {{ queryResult.filter_doc || 'All PDFs' }}
                </span>
              </div>
              <div class="text-xs text-slate-400">
                Latency: <b class="text-indigo-400">{{ queryResult.latency_ms }} ms</b> |
                Device: <b class="uppercase">{{ queryResult.device_used }}</b>
              </div>
            </div>

            <p class="text-sm leading-relaxed text-slate-200 whitespace-pre-line">
              {{ queryResult.answer }}
            </p>
          </div>

          <!-- Retrieved FAISS Chunks Inspector -->
          <div class="p-6 rounded-2xl bg-slate-900 border border-slate-800 space-y-4">
            <div class="flex items-center justify-between">
              <div>
                <h3 class="text-sm font-bold text-white uppercase tracking-wider">
                  📚 Retrieved FAISS Vector Chunks (The "R" in RAG)
                </h3>
                <p class="text-xs text-slate-400 mt-0.5">
                  Each chunk carries its <code class="text-indigo-300">doc_name</code> and <code class="text-indigo-300">page</code> metadata inside FAISS.
                </p>
              </div>
            </div>

            <div class="space-y-3">
              <div
                v-for="(src, index) in queryResult.sources"
                :key="index"
                class="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2"
              >
                <div class="flex flex-wrap items-center justify-between gap-2 text-xs">
                  <div class="flex items-center gap-2">
                    <span class="px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-300 font-semibold">
                      Rank #{{ index + 1 }}
                    </span>
                    <span class="px-2 py-0.5 rounded bg-slate-800 text-indigo-200 font-medium">
                      📘 {{ src.doc_name }}
                    </span>
                    <span class="px-2 py-0.5 rounded bg-slate-800 text-slate-300 font-medium">
                      📄 Page {{ src.page }}
                    </span>
                  </div>
                  <span
                    class="px-2.5 py-0.5 rounded-full font-mono"
                    :class="
                      src.l2_distance < 1.1
                        ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                        : 'bg-amber-500/10 text-amber-400 border border-amber-500/20'
                    "
                  >
                    L2 Distance: {{ src.l2_distance }}
                  </span>
                </div>
                <p class="text-xs text-slate-300 font-mono leading-relaxed bg-slate-900/70 p-3 rounded-lg border border-slate-800/60">
                  {{ src.content }}
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>
    </main>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'

const config = useRuntimeConfig()
const apiBase = config.public.apiBase

interface HealthStatus {
  status: string
  pytorch_version: string
  compute_device: string
  embedding_model: string
  llm_model: string
  index_loaded: boolean
  total_documents: number
  total_chunks_indexed: number
}

interface DocumentInfo {
  filename: string
  pages: number
  chunks: number
  uploaded_at: string
}

interface SourceChunk {
  doc_name: string
  page: number
  content: string
  l2_distance: number
}

interface QueryResponse {
  question: string
  answer: string
  filter_doc: string | null
  sources: SourceChunk[]
  device_used: string
  latency_ms: number
}

const health = ref<HealthStatus | null>(null)
const documents = ref<DocumentInfo[]>([])
const selectedDocFilter = ref<string>('')

const selectedFile = ref<File | null>(null)
const uploading = ref(false)
const uploadResult = ref<any>(null)
const uploadError = ref('')

const question = ref('')
const topK = ref(5)
const querying = ref(false)
const queryResult = ref<QueryResponse | null>(null)
const queryError = ref('')

const sampleQuestions = [
  'How much is the home office stipend and what can I buy with it?',
  'If there is a critical security bug on a Friday, can I deploy to production?',
  'How fast do I need to acknowledge a SEV-1 PagerDuty alert?',
  'What is the company policy on bringing dogs to the office?'
]

async function fetchHealth() {
  try {
    health.value = await $fetch<HealthStatus>(`${apiBase}/api/health`)
  } catch (err) {
    console.error('Failed to connect to FastAPI:', err)
  }
}

async function fetchDocuments() {
  try {
    const res = await $fetch<{ documents: DocumentInfo[] }>(`${apiBase}/api/documents`)
    documents.value = res.documents
    if (selectedDocFilter.value && !documents.value.some(d => d.filename === selectedDocFilter.value)) {
      selectedDocFilter.value = ''
    }
  } catch (err) {
    console.error('Failed to fetch documents:', err)
  }
}

async function refreshAll() {
  await Promise.all([fetchHealth(), fetchDocuments()])
}

function onFileSelected(event: Event) {
  const target = event.target as HTMLInputElement
  if (target.files && target.files.length > 0) {
    selectedFile.value = target.files[0]
    uploadResult.value = null
    uploadError.value = ''
  }
}

async function uploadPdf() {
  if (!selectedFile.value) return
  uploading.value = true
  uploadError.value = ''
  uploadResult.value = null

  try {
    const formData = new FormData()
    formData.append('file', selectedFile.value)

    const res = await $fetch(`${apiBase}/api/documents/upload`, {
      method: 'POST',
      body: formData
    })
    uploadResult.value = res
    await refreshAll()
  } catch (err: any) {
    uploadError.value = err?.data?.detail || err?.message || 'Upload failed'
  } finally {
    uploading.value = false
  }
}

async function deletePdf(filename: string) {
  try {
    await $fetch(`${apiBase}/api/documents/${encodeURIComponent(filename)}`, {
      method: 'DELETE'
    })
    await refreshAll()
  } catch (err: any) {
    alert(err?.data?.detail || 'Failed to delete document')
  }
}

function askPreset(presetQuestion: string) {
  question.value = presetQuestion
  submitQuestion()
}

async function submitQuestion() {
  if (!question.value.trim()) return
  querying.value = true
  queryError.value = ''

  try {
    const res = await $fetch<QueryResponse>(`${apiBase}/api/query`, {
      method: 'POST',
      body: {
        question: question.value,
        top_k: topK.value,
        doc_name: selectedDocFilter.value || null
      }
    })
    queryResult.value = res
  } catch (err: any) {
    queryError.value = err?.data?.detail || err?.message || 'Query failed'
  } finally {
    querying.value = false
  }
}

onMounted(() => {
  refreshAll()
})
</script>
