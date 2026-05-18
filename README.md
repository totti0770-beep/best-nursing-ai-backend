# Best Nursing Practice AI — Backend

FastAPI backend for the BNP Clinical AI clinical nursing assistant.
Powered by Supabase + OpenAI + Pinecone (RAG architecture).

---

## 🚀 النشر على Railway — خطوات سريعة

### 1️⃣ ارفع المشروع إلى GitHub

افتح Terminal داخل مجلد المشروع:

```bash
git init
git add .
git commit -m "Initial commit — BNP backend ready for deploy"
git branch -M main
git remote add origin https://github.com/totti0770-beep/best-nursing-ai-backend.git
git push -u origin main
```

> ⚠️ تأكد أن ملف `.env` **غير** مرفوع (محمي بـ `.gitignore`).

---

### 2️⃣ أنشئ مشروع على Railway

1. اذهب إلى [railway.app](https://railway.app) → سجل دخول بـ GitHub.
2. اضغط **New Project** → **Deploy from GitHub repo**.
3. اختر `best-nursing-ai-backend`.
4. Railway سيكتشف Python تلقائياً ويبدأ البناء.

---

### 3️⃣ أضف متغيرات البيئة (Environment Variables)

في Railway → **Variables** → أضف هذه القيم (احصل عليها من حساباتك):

| Variable | Value |
|---|---|
| `SECRET_KEY` | (مفتاح عشوائي قوي — مثل: `openssl rand -hex 32`) |
| `APP_ENV` | `production` |
| `SUPABASE_URL` | `https://xxxx.supabase.co` |
| `SUPABASE_ANON_KEY` | (من Supabase → Settings → API) |
| `SUPABASE_SERVICE_KEY` | (من Supabase — **سري جداً**) |
| `SUPABASE_BUCKET` | `clinical-docs` |
| `OPENAI_API_KEY` | `sk-...` |
| `PINECONE_API_KEY` | (من Pinecone) |
| `PINECONE_INDEX_NAME` | `nursing-ai` |
| `PINECONE_ENVIRONMENT` | `gcp-starter` |
| `ALLOWED_ORIGINS` | `https://abdullah-gaisy.vercel.app,http://localhost:3000` |

---

### 4️⃣ احصل على رابط الـ API العام

بعد أن ينتهي البناء (3–5 دقائق):

1. Railway → **Settings** → **Networking** → **Generate Domain**.
2. ستحصل على رابط مثل: `https://best-nursing-ai-backend.up.railway.app`.
3. اختبره:

```bash
curl https://your-app.up.railway.app/health
# Expected: {"status":"ok","version":"2.0.0"}
```

افتح `/docs` لرؤية واجهة Swagger التفاعلية:
`https://your-app.up.railway.app/docs`

---

## 🔧 التشغيل محلياً (Local Development)

```bash
# 1. أنشئ بيئة افتراضية
python -m venv venv
source venv/bin/activate   # على Windows: venv\Scripts\activate

# 2. ثبت المتطلبات
pip install -r requirements.txt

# 3. انسخ ملف البيئة
cp .env.example .env
# عدّل .env بمفاتيحك الحقيقية

# 4. شغّل السيرفر
uvicorn main:app --reload
```

افتح: <http://localhost:8000/docs>

---

## 📁 بنية المشروع

```
backend/
├── main.py              # FastAPI entry point
├── requirements.txt     # Python dependencies
├── Procfile             # Railway start command
├── railway.json         # Railway config
├── nixpacks.toml        # Build config (Python 3.11)
├── runtime.txt          # Python version pin
├── .env.example         # Template for env vars
├── .gitignore
│
├── core/
│   ├── config.py        # Settings (Pydantic)
│   ├── auth.py          # JWT auth helpers
│   └── database.py      # Supabase + Pinecone clients
│
├── models/
│   └── schemas.py       # Pydantic request/response models
│
├── routers/
│   ├── auth.py          # /api/v1/auth — login, register
│   ├── users.py         # /api/v1/users
│   ├── documents.py     # /api/v1/documents — upload, list
│   ├── chat.py          # /api/v1/chat — RAG queries
│   └── feedback.py      # /api/v1/feedback
│
└── services/
    ├── document_processor.py  # PDF parsing & chunking
    ├── processing.py          # Embeddings + Pinecone upsert
    └── rag.py                 # Retrieval + generation
```

---

## 🩺 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET`  | `/health` | Health check |
| `GET`  | `/docs` | Swagger UI |
| `POST` | `/api/v1/auth/register` | Create account |
| `POST` | `/api/v1/auth/login` | Get JWT token |
| `POST` | `/api/v1/documents/upload` | Upload PDF (clinical source) |
| `GET`  | `/api/v1/documents/` | List user's documents |
| `POST` | `/api/v1/chat/query` | Ask a clinical question (RAG) |
| `POST` | `/api/v1/feedback/` | Submit feedback on an answer |

---

## ⚠️ ملاحظات أمنية مهمة

1. **لا ترفع `.env` إلى GitHub أبداً** — محمي بـ `.gitignore`.
2. `SUPABASE_SERVICE_KEY` يجب أن يبقى في السيرفر فقط (لا تضعه في React app).
3. غيّر `SECRET_KEY` لقيمة عشوائية في الإنتاج:
   ```bash
   openssl rand -hex 32
   ```
4. ضبط `ALLOWED_ORIGINS` لرابط الـ frontend الفعلي فقط (لا تستخدم `*` في الإنتاج).

---

## 🐛 حل المشاكل الشائعة

**خطأ: Build failed على Railway**
→ تحقق من logs في Railway. عادةً السبب نسخة Python غير صحيحة — تأكد أن `runtime.txt` يحتوي `python-3.11.9`.

**خطأ: 503 Service Unavailable بعد النشر**
→ تحقق من أن جميع متغيرات البيئة مضافة في Railway.

**خطأ: CORS error من الـ frontend**
→ أضف رابط الـ frontend الفعلي في متغير `ALLOWED_ORIGINS` (مفصول بفاصلة).

---

**Contact:** agaissy@moh.gov.sa
**License:** Proprietary — Patent SA-PAT-2026-BNP-001
