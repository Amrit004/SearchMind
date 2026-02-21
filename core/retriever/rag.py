"""
SearchMind RAG Engine
Retrieval-Augmented Generation: retrieve relevant docs, build context, generate answer.
Supports multiple LLM backends (OpenAI, Anthropic, local Ollama).
"""
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, AsyncGenerator
from storage.vector_store import Document
from core.retriever.hybrid import HybridRetriever

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    answer: str
    sources: List[Document]
    query: str
    model: str
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0


class PromptBuilder:
    """Builds RAG prompts from retrieved context."""

    @staticmethod
    def build(query: str, docs: List[Document], system_prompt: str = "") -> str:
        context_parts = []
        for i, doc in enumerate(docs, 1):
            source_label = doc.metadata.get("source", f"Document {i}")
            context_parts.append(f"[{i}] {source_label}\n{doc.text}")

        context = "\n\n---\n\n".join(context_parts)

        default_system = (
            "You are a helpful assistant. Answer the question based ONLY on the provided context. "
            "If the context does not contain enough information to answer the question, say so clearly. "
            "Cite the document numbers [1], [2], etc. when referencing specific information."
        )

        return f"""{system_prompt or default_system}

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""

    @staticmethod
    def build_messages(query: str, docs: List[Document], system_prompt: str = "") -> List[Dict]:
        """Build OpenAI-style messages list."""
        context_parts = []
        for i, doc in enumerate(docs, 1):
            source_label = doc.metadata.get("source", f"Document {i}")
            context_parts.append(f"[{i}] {source_label}\n{doc.text}")

        context = "\n\n---\n\n".join(context_parts)

        system = system_prompt or (
            "You are a helpful assistant. Answer questions based only on the provided context. "
            "Cite source numbers [1], [2] etc. If the context is insufficient, say so clearly."
        )

        user_content = f"Context:\n{context}\n\nQuestion: {query}"

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]


class LLMBackend:
    """Abstract LLM interface supporting multiple providers."""

    async def generate(self, prompt: str, messages: List[Dict] = None, **kwargs) -> str:
        raise NotImplementedError

    async def stream(self, messages: List[Dict], **kwargs) -> AsyncGenerator[str, None]:
        raise NotImplementedError
        yield  # make this a generator


class OpenAIBackend(LLMBackend):
    def __init__(self, model: str = "gpt-4o-mini", api_key: str = ""):
        self.model = model
        self.api_key = api_key

    async def generate(self, prompt: str = "", messages: List[Dict] = None, **kwargs) -> str:
        import openai
        client = openai.AsyncOpenAI(api_key=self.api_key)
        msgs = messages or [{"role": "user", "content": prompt}]
        response = await client.chat.completions.create(
            model=self.model,
            messages=msgs,
            temperature=kwargs.get("temperature", 0.2),
            max_tokens=kwargs.get("max_tokens", 1000),
        )
        return response.choices[0].message.content

    async def stream(self, messages: List[Dict], **kwargs) -> AsyncGenerator[str, None]:
        import openai
        client = openai.AsyncOpenAI(api_key=self.api_key)
        async with client.chat.completions.stream(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", 0.2),
        ) as stream:
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content


class AnthropicBackend(LLMBackend):
    def __init__(self, model: str = "claude-haiku-4-5-20251001", api_key: str = ""):
        self.model = model
        self.api_key = api_key

    async def generate(self, prompt: str = "", messages: List[Dict] = None, **kwargs) -> str:
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=self.api_key)
        msgs = messages or [{"role": "user", "content": prompt}]
        system = next((m["content"] for m in msgs if m["role"] == "system"), "")
        user_msgs = [m for m in msgs if m["role"] != "system"]

        response = await client.messages.create(
            model=self.model,
            max_tokens=kwargs.get("max_tokens", 1000),
            system=system,
            messages=user_msgs,
        )
        return response.content[0].text


class OllamaBackend(LLMBackend):
    """Local LLM via Ollama (llama3, mistral, etc.)."""

    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    async def generate(self, prompt: str = "", messages: List[Dict] = None, **kwargs) -> str:
        import aiohttp
        msgs = messages or [{"role": "user", "content": prompt}]
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/chat",
                json={"model": self.model, "messages": msgs, "stream": False},
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                data = await resp.json()
                return data["message"]["content"]

    async def stream(self, messages: List[Dict], **kwargs) -> AsyncGenerator[str, None]:
        import aiohttp, json
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/chat",
                json={"model": self.model, "messages": messages, "stream": True},
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                async for line in resp.content:
                    if line:
                        data = json.loads(line)
                        if content := data.get("message", {}).get("content"):
                            yield content


class RAGEngine:
    """
    Complete RAG pipeline: retrieve → build context → generate → cite sources.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        llm_backend: Optional[LLMBackend] = None,
        top_k: int = 5,
        alpha: float = 0.6,
    ):
        self.retriever = retriever
        self.llm = llm_backend or OllamaBackend()  # Default to local Ollama
        self.top_k = top_k
        self.alpha = alpha

    async def query(
        self,
        collection: str,
        question: str,
        top_k: Optional[int] = None,
        use_reranker: bool = False,
        system_prompt: str = "",
        filter_conditions: Optional[Dict] = None,
    ) -> RAGResponse:
        """Full RAG query: retrieve → generate → return with sources."""
        import time

        k = top_k or self.top_k

        # Step 1: Retrieve relevant documents
        t0 = time.time()
        docs = await self.retriever.search(
            collection=collection,
            query=question,
            top_k=k,
            alpha=self.alpha,
            filter_conditions=filter_conditions,
            use_reranker=use_reranker,
        )
        retrieval_ms = (time.time() - t0) * 1000

        if not docs:
            return RAGResponse(
                answer="I could not find relevant information to answer this question.",
                sources=[],
                query=question,
                model=str(type(self.llm).__name__),
                retrieval_ms=retrieval_ms,
            )

        # Step 2: Build context prompt
        messages = PromptBuilder.build_messages(question, docs, system_prompt)

        # Step 3: Generate answer
        t1 = time.time()
        try:
            answer = await self.llm.generate(messages=messages)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            answer = f"Generation failed: {e}. Retrieved {len(docs)} documents."
        generation_ms = (time.time() - t1) * 1000

        return RAGResponse(
            answer=answer,
            sources=docs,
            query=question,
            model=str(type(self.llm).__name__),
            retrieval_ms=round(retrieval_ms, 1),
            generation_ms=round(generation_ms, 1),
        )

    async def stream_query(
        self, collection: str, question: str, top_k: Optional[int] = None
    ) -> AsyncGenerator[str, None]:
        """Stream RAG response token by token."""
        k = top_k or self.top_k
        docs = await self.retriever.search(collection, question, k, self.alpha)
        messages = PromptBuilder.build_messages(question, docs)

        yield f"data: {{'type': 'sources', 'count': {len(docs)}}}\n\n"

        async for token in self.llm.stream(messages):
            yield f"data: {{'type': 'token', 'content': {repr(token)}}}\n\n"

        yield "data: {'type': 'done'}\n\n"
