"""
ProviderRegistry — maps provider names to client factory callables.

Design
------
- Abstract Factory pattern: callers ask for a client by (provider, modality)
  and get back a fully configured instance.
- Factories are plain callables ``(config: ClientConfig) -> BaseClient``,
  making it easy to register lambdas, class constructors, or factory functions.
- Thread-safe; supports a global singleton alongside isolated instances.

Usage
-----
::

    from conduit_sdk.registry import ProviderRegistry
    from conduit_sdk.clients import LLMClient

    registry = ProviderRegistry.global_registry()

    # Register your provider client
    registry.register_factory(
        provider="openai",
        modality="llm",
        factory=lambda cfg: MyOpenAIClient(config=cfg),
    )

    # Resolve a client at runtime
    client = registry.create_client(
        provider="openai",
        modality="llm",
        config=ClientConfig(model="gpt-4o", api_key="sk-..."),
    )
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import ClassVar

from conduit_sdk.core.config import ClientConfig
from conduit_sdk.registry.model_registry import Modality

# Type alias for factory functions
ClientFactory = Callable[[ClientConfig], object]  # returns a BaseClient subclass


class ProviderRegistry:
    """
    Thread-safe registry mapping (provider, modality) → client factory.

    A global singleton is available via ``ProviderRegistry.global_registry()``.
    """

    _global: ClassVar[ProviderRegistry | None] = None
    _global_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._factories: dict[tuple[str, Modality], ClientFactory] = {}

    # ------------------------------------------------------------------
    # Global singleton
    # ------------------------------------------------------------------

    @classmethod
    def global_registry(cls) -> ProviderRegistry:
        """Return the process-wide singleton registry."""
        if cls._global is None:
            with cls._global_lock:
                if cls._global is None:
                    cls._global = cls()
        return cls._global

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_factory(
        self,
        provider: str,
        modality: Modality,
        factory: ClientFactory,
    ) -> ProviderRegistry:
        """
        Register a factory callable for a (provider, modality) pair.

        Parameters
        ----------
        provider:
            Provider identifier (e.g. ``"openai"``).
        modality:
            One of ``"llm"``, ``"image"``, ``"video"``, ``"embedding"``.
        factory:
            Callable ``(config: ClientConfig) -> BaseClient`` that constructs
            and returns a configured client.

        Returns
        -------
        ProviderRegistry
            Self, for fluent chaining.
        """
        with self._lock:
            self._factories[(provider, modality)] = factory
        return self

    # ------------------------------------------------------------------
    # Resolution / instantiation
    # ------------------------------------------------------------------

    def create_client(
        self,
        provider: str,
        modality: Modality,
        config: ClientConfig | None = None,
    ) -> object:
        """
        Instantiate a client via the registered factory.

        Parameters
        ----------
        provider:
            Provider name to look up.
        modality:
            Desired modality.
        config:
            Config to pass to the factory.  Defaults to ``ClientConfig()``.

        Returns
        -------
        BaseClient
            A fully constructed client instance.

        Raises
        ------
        RegistryError
            If no factory is registered for the (provider, modality) pair.
        """
        from conduit_sdk.core.exceptions import RegistryError

        key = (provider, modality)
        with self._lock:
            factory = self._factories.get(key)

        if factory is None:
            available = sorted(f"{p}/{m}" for p, m in self._factories)
            raise RegistryError(
                f"No factory registered for provider={provider!r}, modality={modality!r}. "
                f"Available: {available}"
            )

        return factory(config or ClientConfig(provider=provider))

    def list_providers(self, *, modality: Modality | None = None) -> list[tuple[str, Modality]]:
        """Return registered (provider, modality) pairs, optionally filtered."""
        with self._lock:
            keys = list(self._factories.keys())
        if modality:
            keys = [(p, m) for p, m in keys if m == modality]
        return sorted(keys)

    def __contains__(self, key: tuple[str, Modality]) -> bool:
        with self._lock:
            return key in self._factories

    def __repr__(self) -> str:
        with self._lock:
            n = len(self._factories)
        return f"ProviderRegistry(factories={n})"
