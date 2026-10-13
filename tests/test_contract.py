from convcontract.contract import ACTIVE, REMOVED, SUPERSEDED, Contract


def ops(**kw):
    return {"goal": None, "add": [], "supersede": [], "remove": [], "material": [], **kw}


def test_add_assigns_ids_and_turn_provenance():
    c = Contract()
    c.apply(ops(goal="Write a SQL query", add=[{"text": "Rank customers by total order value", "quote": "by total order value"}]))
    c.apply(ops(add=[{"text": "Only 2025 orders", "quote": "last year only"}]))
    assert [(r.id, r.turn) for r in c.requirements] == [("R1", 1), ("R2", 2)]
    assert c.goal == "Write a SQL query"
    assert c.requirements[1].quote == "last year only"


def test_supersede_replaces_and_hides_old_requirement():
    c = Contract()
    c.apply(ops(add=[{"text": "Return the top 10 customers", "quote": "top 10"}]))
    warnings = c.apply(ops(supersede=[{"id": "R1", "text": "Return the top 5 customers", "quote": "just top 5"}]))
    assert warnings == []
    old, new = c.requirements
    assert (old.status, old.superseded_by) == (SUPERSEDED, "R2")
    assert (new.status, new.turn) == (ACTIVE, 2)
    task = c.render_task()
    assert "top 5" in task and "top 10" not in task


def test_remove_and_unknown_ids_warn():
    c = Contract()
    c.apply(ops(add=[{"text": "A", "quote": "a"}, {"text": "B", "quote": "b"}]))
    warnings = c.apply(ops(remove=["R1", "R9"], supersede=[{"id": "R7", "text": "C", "quote": "c"}]))
    assert c.requirements[0].status == REMOVED
    assert len(warnings) == 2
    # a supersede with an unknown id still keeps the user's new information
    assert [r.text for r in c.active()] == ["B", "C"]


def test_open_questions_replaced_only_when_present():
    c = Contract()
    c.apply(ops(open_questions=["Which year?"]))
    c.apply(ops())
    assert c.open_questions == ["Which year?"]
    c.apply(ops(open_questions=[]))
    assert c.open_questions == []


def test_render_includes_material_and_draft_only_on_request():
    c = Contract()
    c.apply(ops(goal="Fix the function", material=["def f(x):\n    return x"]))
    c.draft = "def f(x):\n    return x + 1"
    assert '<material id="M1">\ndef f(x):\n    return x\n</material>' in c.render_task()
    assert "previous_answer" not in c.render_task()
    assert "return x + 1" in c.render_task(include_draft=True)


def test_view_for_extractor_lists_only_active():
    c = Contract()
    c.apply(ops(add=[{"text": "A", "quote": "a"}]))
    c.apply(ops(supersede=[{"id": "R1", "text": "A2", "quote": "a2"}]))
    assert '"R1"' not in c.view_for_extractor() and '"R2"' in c.view_for_extractor()


def test_round_trip():
    c = Contract()
    c.apply(ops(goal="g", add=[{"text": "A", "quote": "a"}], material=["m"], open_questions=["q"]))
    c.draft = "d"
    assert Contract.from_dict(c.to_dict()) == c


def test_render_lists_open_questions():
    c = Contract()
    c.apply(ops(add=[{"text": "Compute total points", "quote": "total points"}], open_questions=["How many games per year?"]))
    assert "Still unknown (the user has not said):\n- How many games per year?" in c.render_task()
    c.apply(ops(open_questions=[]))
    assert "Still unknown" not in c.render_task()
