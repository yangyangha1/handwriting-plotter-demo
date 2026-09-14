from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path

IDS_ARITY = {
    "⿰": 2,
    "⿱": 2,
    "⿴": 2,
    "⿵": 2,
    "⿶": 2,
    "⿷": 2,
    "⿸": 2,
    "⿹": 2,
    "⿺": 2,
    "⿻": 2,
    "⿲": 3,
    "⿳": 3,
}
IDS_OPERATORS = set(IDS_ARITY)
PLACEHOLDERS = set("？?*〾")


@dataclass(frozen=True, slots=True)
class IDSNode:
    symbol: str
    children: tuple[IDSNode, ...] = ()

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def at_path(self, path: tuple[int, ...] | list[int]) -> IDSNode | None:
        node: IDSNode = self
        for index in path:
            if index < 0 or index >= len(node.children):
                return None
            node = node.children[index]
        return node

    def serialize(self) -> str:
        if not self.children:
            return self.symbol
        return self.symbol + "".join(child.serialize() for child in self.children)

    def leaves(self) -> tuple[str, ...]:
        if not self.children:
            return () if self.symbol in PLACEHOLDERS else (self.symbol,)
        result: list[str] = []
        for child in self.children:
            result.extend(child.leaves())
        return tuple(result)


@dataclass(frozen=True, slots=True)
class CharacterComponents:
    char: str
    decomposition: str
    radical: str | None
    components: frozenset[str]
    matches: tuple[tuple[int, ...] | None, ...]
    stroke_components: tuple[str | None, ...]
    component_strokes: dict[str, tuple[int, ...]]
    decomposition_valid: bool = True


def _parse_ids_node(text: str, pos: int = 0) -> tuple[IDSNode, int]:
    if pos >= len(text):
        raise ValueError("unexpected end of IDS")
    symbol = text[pos]
    pos += 1
    arity = IDS_ARITY.get(symbol, 0)
    if not arity:
        return IDSNode(symbol), pos
    children: list[IDSNode] = []
    for _ in range(arity):
        child, pos = _parse_ids_node(text, pos)
        children.append(child)
    return IDSNode(symbol, tuple(children)), pos


def parse_ids(decomposition: str) -> IDSNode | None:
    """Parse a Make Me A Hanzi IDS decomposition into a tree.

    Returns None for explicitly unknown decompositions.  MMH uses full-width
    question marks for unavailable/partial data; partial trees are still parsed
    so known branches remain useful for stroke ownership.
    """
    text = decomposition.strip()
    if not text or text[0] in PLACEHOLDERS:
        return None
    try:
        root, pos = _parse_ids_node(text, 0)
    except ValueError:
        return None
    # Variation selectors occasionally follow an ideograph.  We accept only
    # harmless trailing whitespace; malformed remainder makes alignment unsafe.
    if text[pos:].strip():
        return None
    return root


def extract_components(decomposition: str, char: str | None = None) -> frozenset[str]:
    root = parse_ids(decomposition)
    if root is None:
        return frozenset()
    out: set[str] = set()
    for token in root.leaves():
        if token in PLACEHOLDERS or token.isspace() or token == char:
            continue
        category = unicodedata.category(token)
        if category.startswith("L") or "CJK" in unicodedata.name(token, ""):
            out.add(token)
    return frozenset(out)


def _normalize_matches(value) -> tuple[tuple[int, ...] | None, ...]:
    if not isinstance(value, list):
        return ()
    result: list[tuple[int, ...] | None] = []
    for entry in value:
        if entry is None:
            result.append(None)
            continue
        if not isinstance(entry, list) or not all(isinstance(i, int) for i in entry):
            result.append(None)
            continue
        result.append(tuple(entry))
    return tuple(result)


def resolve_stroke_components(
    decomposition: str,
    matches: tuple[tuple[int, ...] | None, ...],
) -> tuple[tuple[str | None, ...], dict[str, tuple[int, ...]]]:
    """Resolve MMH stroke ``matches`` paths to concrete component symbols.

    Each matches entry corresponds to one character stroke and stores a path in
    the IDS decomposition tree.  The result therefore provides exact
    component-to-stroke ownership wherever MMH supplies a valid match.
    """
    root = parse_ids(decomposition)
    if root is None:
        return tuple(None for _ in matches), {}

    owners: list[str | None] = []
    grouped: dict[str, list[int]] = {}
    for stroke_index, path in enumerate(matches):
        if path is None:
            owners.append(None)
            continue
        node = root.at_path(path)
        if node is None:
            owners.append(None)
            continue
        # MMH normally targets a concrete component node.  If a path points to
        # a subtree, keep its serialized IDS so ownership is still deterministic
        # without pretending it is a single radical.
        label = node.symbol if node.is_leaf else node.serialize()
        if label in PLACEHOLDERS or label in IDS_OPERATORS:
            owners.append(None)
            continue
        owners.append(label)
        grouped.setdefault(label, []).append(stroke_index)
    return tuple(owners), {key: tuple(value) for key, value in grouped.items()}


class MMHDictionarySource:
    """Read Make Me A Hanzi ``dictionary.txt`` JSON-lines data.

    In addition to decomposition semantics, MMH's ``matches`` field explicitly
    maps each character stroke to a node in the decomposition tree.  This class
    resolves those paths once and exposes exact component-owned stroke indices
    for training and synthesis.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._index: dict[str, CharacterComponents] | None = None

    def _load(self) -> None:
        index: dict[str, CharacterComponents] = {}
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                char = str(obj.get("character", ""))
                if not char:
                    continue
                decomp = str(obj.get("decomposition", ""))
                radical = obj.get("radical")
                components = set(extract_components(decomp, char))
                if isinstance(radical, str) and radical and radical != char:
                    components.add(radical)
                matches = _normalize_matches(obj.get("matches"))
                owners, component_strokes = resolve_stroke_components(decomp, matches)
                index[char] = CharacterComponents(
                    char=char,
                    decomposition=decomp,
                    radical=radical if isinstance(radical, str) else None,
                    components=frozenset(components),
                    matches=matches,
                    stroke_components=owners,
                    component_strokes=component_strokes,
                    decomposition_valid=parse_ids(decomp) is not None,
                )
        self._index = index

    def get(self, char: str) -> CharacterComponents | None:
        if self._index is None:
            self._load()
        assert self._index is not None
        return self._index.get(char)

    def characters(self) -> list[str]:
        if self._index is None:
            self._load()
        assert self._index is not None
        return list(self._index)
