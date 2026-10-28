-- ============================================================================
-- StudyRAG — Supabase Database Schema (PostgreSQL)
-- ============================================================================
-- Run this script in your Supabase Project's SQL Editor to enable cloud document
-- metadata tracking, user profiles, and row-level security (RLS).
-- ============================================================================

-- 1. Create User Documents Table
CREATE TABLE IF NOT EXISTS public.user_documents (
    id TEXT PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
    filename TEXT NOT NULL,
    file_size_bytes BIGINT NOT NULL,
    total_pages INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- 2. Create User Quiz History Table
CREATE TABLE IF NOT EXISTS public.user_quizzes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
    difficulty TEXT NOT NULL,
    total_questions INTEGER NOT NULL,
    score_percentage NUMERIC(5, 2) NOT NULL,
    grade TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- 3. Enable Row-Level Security (RLS)
ALTER TABLE public.user_documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.user_quizzes ENABLE ROW LEVEL SECURITY;

-- 4. Policies for user_documents
CREATE POLICY "Users can only view their own documents"
    ON public.user_documents FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own documents"
    ON public.user_documents FOR INSERT
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete their own documents"
    ON public.user_documents FOR DELETE
    USING (auth.uid() = user_id);

-- 5. Policies for user_quizzes
CREATE POLICY "Users can only view their own quizzes"
    ON public.user_quizzes FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own quizzes"
    ON public.user_quizzes FOR INSERT
    WITH CHECK (auth.uid() = user_id);
