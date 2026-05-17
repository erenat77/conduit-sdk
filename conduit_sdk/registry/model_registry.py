"""
ModelRegistry — a central catalogue of named model definitions.

Design
------
- Registry pattern: maps string keys → ``ModelDefinition`` value objects.
- Thread-safe for concurrent reads; writes are protected by a lock.
- Supports namespaced keys (``"openai/gpt-4o"``) and unqualified aliases
  (``"gpt-4o"``).
- Singleton-per-name via the class-level ``_instances`` dict; callers can
  also construct isolated ``ModelRegistry`` instances for testing.

Usage
-----
::

    registry = ModelRegistry.global_registry()

    registry.register(ModelDefinition(
        name="openai/gpt-4o",
        provider="openai",
        modality="llm",
        context_window=128_000,
        max_output_tokens=4096,
        aliases=["gpt-4o"],
    ))

    definition = registry.resolve("gpt-4o")
    print(definition.provider)  # "openai"
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

Modality = Literal["llm", "image", "video", "embedding"]


@dataclass(frozen=True)
class ModelDefinition:
    """
    Immutable description of a model available in a provider.

    Parameters
    ----------
    name:
        Canonical qualified name, e.g. ``"openai/gpt-4o"``.
    provider:
        Provider identifier, e.g. ``"openai"``.
    modality:
        One of ``"llm"``, ``"image"``, ``"video"``, ``"embedding"``.
    aliases:
        Alternative short names that resolve to this definition.
    context_window:
        LLM-specific: max input context in tokens.
    max_output_tokens:
        LLM-specific: max generated tokens.
    input_cost_per_1k_tokens:
        Pricing hint (USD); used by CostCalculator if not overridden.
    output_cost_per_1k_tokens:
        Pricing hint (USD).
    metadata:
        Arbitrary extra fields (e.g. ``{"supports_vision": True}``).
    """

    name: str
    provider: str
    modality: Modality
    aliases: list[str] = field(default_factory=list)
    context_window: int | None = None
    max_output_tokens: int | None = None
    input_cost_per_1k_tokens: float | None = None
    output_cost_per_1k_tokens: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelRegistry:
    """
    Thread-safe registry mapping model names and aliases to ``ModelDefinition``s.

    A global singleton registry is available via ``ModelRegistry.global_registry()``.
    Create isolated instances for testing or multi-tenant scenarios.
    """

    _global: ClassVar[ModelRegistry | None] = None
    _global_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # Both canonical names and aliases map to the same definition
        self._registry: dict[str, ModelDefinition] = {}

    # ------------------------------------------------------------------
    # Global singleton
    # ------------------------------------------------------------------

    @classmethod
    def global_registry(cls) -> ModelRegistry:
        """Return the process-wide singleton registry."""
        if cls._global is None:
            with cls._global_lock:
                if cls._global is None:
                    cls._global = cls()
        return cls._global

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, definition: ModelDefinition) -> ModelRegistry:
        """
        Register a model definition (and its aliases).

        Re-registering an existing name silently overwrites it.

        Parameters
        ----------
        definition:
            The ``ModelDefinition`` to register.

        Returns
        -------
        ModelRegistry
            Self, for fluent chaining.
        """
        with self._lock:
            self._registry[definition.name] = definition
            for alias in definition.aliases:
                self._registry[alias] = definition
        return self

    def register_many(self, definitions: list[ModelDefinition]) -> ModelRegistry:
        """Register a batch of definitions atomically."""
        with self._lock:
            for defn in definitions:
                self.register(defn)
        return self

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def resolve(self, name: str) -> ModelDefinition:
        """
        Look up a model by canonical name or alias.

        Raises
        ------
        RegistryError
            If ``name`` is not found.
        """
        from conduit_sdk.core.exceptions import RegistryError

        with self._lock:
            if name not in self._registry:
                available = sorted(set(self._registry.keys()))
                raise RegistryError(f"Model {name!r} not found in registry. Available: {available}")
            return self._registry[name]

    def get(self, name: str) -> ModelDefinition | None:
        """Like ``resolve`` but returns ``None`` instead of raising."""
        with self._lock:
            return self._registry.get(name)

    def list_models(self, *, modality: Modality | None = None) -> list[ModelDefinition]:
        """Return unique definitions, optionally filtered by modality."""
        with self._lock:
            seen: set[str] = set()
            result: list[ModelDefinition] = []
            for defn in self._registry.values():
                if defn.name in seen:
                    continue
                if modality is None or defn.modality == modality:
                    result.append(defn)
                    seen.add(defn.name)
            return sorted(result, key=lambda d: d.name)

    def __contains__(self, name: str) -> bool:
        with self._lock:
            return name in self._registry

    def __len__(self) -> int:
        with self._lock:
            return len({d.name for d in self._registry.values()})

    def __repr__(self) -> str:
        return f"ModelRegistry(models={len(self)})"
