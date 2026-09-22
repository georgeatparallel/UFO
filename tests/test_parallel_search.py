# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import asyncio
from unittest.mock import AsyncMock

import httpx
from fastmcp import FastMCP
from fastmcp.client.transports import StreamableHttpTransport
from langchain.docstore.document import Document

from config.config_loader import ConfigLoader
from ufo.rag import retriever, web_search
from ufo.rag.web_search import ParallelSearchWeb


def test_parallel_provider_loads_from_rag_yaml(tmp_path):
    config_dir = tmp_path / "config" / "ufo"
    config_dir.mkdir(parents=True)
    (config_dir / "rag.yaml").write_text(
        'RAG_ONLINE_SEARCH: true\nRAG_ONLINE_SEARCH_PROVIDER: "parallel"\n'
    )

    config = ConfigLoader(base_path=str(tmp_path / "config")).load_ufo_config()

    assert config.rag.online_search is True
    assert config.rag.online_search_provider == "parallel"


def test_parallel_search_invokes_web_search_and_surfaces_evidence():
    server = FastMCP("parallel-search-test")
    calls = []

    @server.tool()
    def web_search(objective: str, search_queries: list[str]):
        calls.append((objective, search_queries))
        return {
            "search_id": "test-search",
            "results": [
                {
                    "title": "UFO documentation",
                    "url": "https://example.com/ufo",
                    "excerpts": ["Useful attributed evidence."],
                }
            ],
            "session_id": "test-session",
        }

    search = ParallelSearchWeb()
    search.transport = server

    results = search.search("UFO agent framework", top_k=1)

    assert calls == [
        (
            "Find current, reliable information about: UFO agent framework",
            ["UFO agent framework"],
        )
    ]
    assert results == [
        {
            "name": "UFO documentation",
            "url": "https://example.com/ufo",
            "snippet": "Useful attributed evidence.",
        }
    ]
    assert search.create_documents(results) == [
        Document(
            page_content="Useful attributed evidence.",
            metadata={
                "name": "UFO documentation",
                "url": "https://example.com/ufo",
                "snippet": "Useful attributed evidence.",
            },
        )
    ]


def test_parallel_transport_sends_ufo_user_agent():
    server = FastMCP("parallel-search-user-agent-test")
    observed_user_agents = []

    @server.tool()
    def web_search(objective: str, search_queries: list[str]):
        return {"results": []}

    app = server.http_app(path="/mcp", stateless_http=True)

    async def capture_headers(scope, receive, send):
        if scope["type"] == "http":
            headers = dict(scope["headers"])
            observed_user_agents.append(headers.get(b"user-agent", b"").decode())
        await app(scope, receive, send)

    def client_factory(headers=None, timeout=None, auth=None):
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=capture_headers),
            base_url="http://parallel.test",
            headers=headers,
            timeout=timeout,
            auth=auth,
        )

    search = ParallelSearchWeb("http://parallel.test/mcp")
    search.transport.httpx_client_factory = client_factory

    async def run_search():
        async with app.lifespan(app):
            return await search._search("UFO")

    assert asyncio.run(run_search()) == []
    assert isinstance(search.transport, StreamableHttpTransport)
    assert observed_user_agents
    assert set(observed_user_agents) == {"UFO"}


def test_online_retriever_keeps_bing_as_default(monkeypatch):
    calls = []

    class FakeBing:
        def search(self, query, top_k):
            calls.append(("bing", query, top_k))
            return [{"name": "Bing", "url": "https://bing.test", "snippet": "Bing"}]

        def create_documents(self, results):
            return [Document(page_content="Bing")]

        def create_indexer(self, documents):
            return "bing-index"

    class UnexpectedParallel:
        def __init__(self):
            raise AssertionError("the default route must not contact Parallel")

    monkeypatch.setattr(web_search, "BingSearchWeb", FakeBing)
    monkeypatch.setattr(web_search, "ParallelSearchWeb", UnexpectedParallel)
    monkeypatch.setattr(
        web_search.ufo_config.rag, "online_search_provider", "bing"
    )

    online = retriever.OnlineDocRetriever("current UFO docs", top_k=2)

    assert online.indexer == "bing-index"
    assert calls == [("bing", "current UFO docs", 2)]


def test_online_retriever_routes_explicit_parallel_selection(monkeypatch):
    calls = []

    class FakeParallel:
        def search(self, query, top_k):
            calls.append(("parallel", query, top_k))
            return [
                {
                    "name": "Parallel",
                    "url": "https://parallel.test",
                    "snippet": "Parallel",
                }
            ]

        def create_documents(self, results):
            return [Document(page_content="Parallel")]

        def create_indexer(self, documents):
            return "parallel-index"

    monkeypatch.setattr(web_search, "ParallelSearchWeb", FakeParallel)
    monkeypatch.setattr(
        web_search.ufo_config.rag, "online_search_provider", "parallel"
    )

    online = retriever.OnlineDocRetriever("current UFO docs", top_k=2)

    assert online.indexer == "parallel-index"
    assert calls == [("parallel", "current UFO docs", 2)]


def test_app_agent_online_search_uses_parallel_from_async_context(monkeypatch):
    from ufo.agents.agent.app_agent import AppAgent

    server = FastMCP("parallel-search-app-agent-test")
    calls = []

    @server.tool(name="web_search")
    def handle_search(objective: str, search_queries: list[str]):
        calls.append(search_queries)
        return {
            "results": [
                {
                    "title": "Current UFO guide",
                    "url": "https://example.com/guide",
                    "excerpts": ["Instructions for the requested task."],
                }
            ]
        }

    class LocalParallelSearchWeb(ParallelSearchWeb):
        def __init__(self):
            self.transport = server

    monkeypatch.setattr(web_search, "ParallelSearchWeb", LocalParallelSearchWeb)
    monkeypatch.setattr(web_search.ufo_config.rag, "online_search", True)
    monkeypatch.setattr(web_search.ufo_config.rag, "online_search_provider", "parallel")
    monkeypatch.setattr(web_search.ufo_config.rag, "online_search_topk", 1)
    monkeypatch.setattr(web_search, "get_hugginface_embedding", lambda: object())
    monkeypatch.setattr(
        web_search.FAISS, "from_documents", lambda documents, embeddings: documents
    )

    agent = AppAgent.__new__(AppAgent)
    agent.retriever_factory = retriever.RetrieverFactory()
    agent._load_mcp_context = AsyncMock()

    asyncio.run(agent.context_provision("UFO docs"))

    assert calls == [["UFO docs"]]
    assert agent.online_doc_retriever.indexer == [
        Document(
            page_content="Instructions for the requested task.",
            metadata={
                "url": "https://example.com/guide",
                "name": "Current UFO guide",
                "snippet": "Instructions for the requested task.",
            },
        )
    ]
