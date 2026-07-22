# Problem Statement - Cancer Clinical AI Evaluation Platform

## Context
Cancer treatments (chemotherapy protocols, radiotherapy dose limits, diagnostic thresholds) require extreme precision. A small error in a clinical response can result in toxicities, under-treatment, or patient harm. When hospitals deploy Retrieval-Augmented Generation (RAG) assistants to help clinicians review guidelines, the underlying systems are frequently upgraded.

## Core Problem
Updates to the following RAG components introduce significant risk:
- **LLM/Embeddings Swap**: New models can alter how semantic queries retrieve chunks, occasionally omitting relevant guidelines.
- **Prompt modifications**: Alterations can cause models to ignore negative constraints, extrapolate patient dosages, or fail to mention citation sources.
- **Knowledge Base growth**: Ingesting new guidelines can introduce conflicting facts or degrade retriever hits for existing test cases.

Without a structured QA platform, there is no verification gate to catch these regressions before they reach clinical staff. This platform solves this challenge by serving as an automated quality gate comparing baseline and candidate pipelines against a golden medical dataset.
