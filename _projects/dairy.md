---
layout: page
title: dAIry — On-Device RAG Diary
description: A private, offline-first Android diary that lets you chat with your own past entries using a local Gemma model. Nothing leaves the phone.
img:
importance: 2
category: fun
related_publications:
---

[GitHub Repository](https://github.com/YashJayswal24/dAIry)

**Summary:**
A privacy-first Android journaling app with a fully on-device RAG pipeline — write entries, browse by calendar, and chat with an on-device Gemma model grounded in your own past entries.

**Highlights:**

- Built an on-device RAG pipeline (MediaPipe Text Embedder + brute-force cosine retrieval + MediaPipe/AICore Gemma inference) so entries, embeddings, and questions never leave the phone.
- Shipped a working app with CI-built debug/release APKs on every push, verified via a 15-prompt judged chat-eval suite run on-device.
