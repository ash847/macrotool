"""Saved conversations and the trade ideas they work on.

Phase 1 (option A): one conversation has one *active* idea — the last view the PM
ran a pack on — which labels the chat. Every idea a conversation touched is recorded
(``conversation_ideas``) so option C (ideas shared across conversations) needs no
schema change, only UI.

Layers, mirroring the rest of the codebase:
- ``models``    pure dataclasses (no IO)
- ``identity``  the one rule for "same idea / new version / new idea"
- ``serialize`` provider message blocks <-> JSON
- ``store``     ``ConversationStore`` seam (Supabase + in-memory)
- ``service``   orchestration: record an exchange, resume, refresh
"""
