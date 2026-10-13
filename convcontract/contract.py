"""The contract: a structured record of what the user has asked for, rebuilt into a single-turn task.

The extractor model proposes edits ("ops") each turn; `Contract.apply` executes them in code, so IDs,
turn provenance and supersession are bookkeeping we control rather than something the model rewrites.
"""
import json
from dataclasses import asdict, dataclass, field

ACTIVE, SUPERSEDED, REMOVED = "active", "superseded", "removed"


@dataclass
class Requirement:
    id: str
    text: str  # standalone restatement, readable without the conversation
    quote: str  # the user's own words
    turn: int  # user turn that introduced it
    status: str = ACTIVE
    superseded_by: str | None = None


@dataclass
class Material:
    id: str
    content: str  # verbatim content the user supplied (code, data, examples)
    turn: int


@dataclass
class Contract:
    goal: str = ""
    requirements: list[Requirement] = field(default_factory=list)
    material: list[Material] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    turn: int = 0
    draft: str | None = None  # latest executor output
    log: list[dict] = field(default_factory=list)  # per-turn ops, for analysis

    def active(self) -> list[Requirement]:
        return [r for r in self.requirements if r.status == ACTIVE]

    def _get(self, req_id: str) -> Requirement | None:
        return next((r for r in self.requirements if r.id == req_id), None)

    def _add(self, text: str, quote: str) -> Requirement:
        req = Requirement(id=f"R{len(self.requirements) + 1}", text=text.strip(), quote=(quote or text).strip(), turn=self.turn)
        self.requirements.append(req)
        return req

    def apply(self, ops: dict) -> list[str]:
        """Apply one turn of extractor ops. Returns warnings for ops that referenced unknown IDs."""
        self.turn += 1
        warnings = []

        if ops.get("goal"):
            self.goal = ops["goal"].strip()

        for item in ops.get("add") or []:
            if item.get("text"):
                self._add(item["text"], item.get("quote", ""))

        for item in ops.get("supersede") or []:
            old = self._get(item.get("id", ""))
            if not item.get("text"):
                continue
            new = self._add(item["text"], item.get("quote", ""))
            if old is None or old.status != ACTIVE:
                warnings.append(f"supersede: unknown or inactive id {item.get('id')!r}; added as new requirement {new.id}")
                continue
            old.status, old.superseded_by = SUPERSEDED, new.id

        for req_id in ops.get("remove") or []:
            old = self._get(req_id)
            if old is None or old.status != ACTIVE:
                warnings.append(f"remove: unknown or inactive id {req_id!r}")
                continue
            old.status = REMOVED

        for content in ops.get("material") or []:
            if content and content.strip():
                self.material.append(Material(id=f"M{len(self.material) + 1}", content=content, turn=self.turn))

        if "open_questions" in ops:
            self.open_questions = [q for q in ops.get("open_questions") or [] if q]

        self.log.append({"turn": self.turn, "ops": ops, "warnings": warnings})
        return warnings

    def view_for_extractor(self) -> str:
        """Compact JSON the extractor edits against (only live entries, with IDs)."""
        return json.dumps(
            {
                "goal": self.goal,
                "requirements": [{"id": r.id, "text": r.text} for r in self.active()],
                "material_ids": [m.id for m in self.material],
                "open_questions": self.open_questions,
            },
            indent=2,
        )

    def render_task(self, include_draft: bool = False) -> str:
        """The single-turn task description the executor sees in place of the conversation."""
        parts = []
        if self.goal:
            parts.append(f"Goal: {self.goal}")
        active = self.active()
        if active:
            parts.append("Requirements (all must be satisfied):\n" + "\n".join(f"{i}. {r.text}" for i, r in enumerate(active, 1)))
        for m in self.material:
            parts.append(f'Material provided by the user ({m.id}, verbatim):\n<material id="{m.id}">\n{m.content}\n</material>')
        if self.open_questions:
            parts.append("Still unknown (the user has not said):\n" + "\n".join(f"- {q}" for q in self.open_questions))
        if include_draft and self.draft:
            parts.append(
                "The user is referring to your previous answer, shown below. Revise it so it satisfies every requirement above; "
                "drop anything in it that conflicts with them.\n<previous_answer>\n" + self.draft + "\n</previous_answer>"
            )
        return "\n\n".join(parts)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Contract":
        return cls(
            goal=d.get("goal", ""),
            requirements=[Requirement(**r) for r in d.get("requirements", [])],
            material=[Material(**m) for m in d.get("material", [])],
            open_questions=list(d.get("open_questions", [])),
            turn=d.get("turn", 0),
            draft=d.get("draft"),
            log=list(d.get("log", [])),
        )
