"""Sharded conversation simulator with the contract layer as the assistant.

This mirrors the harness's `ConversationSimulatorSharded.run` loop line for line (same simulated user,
same turn classifier, same answer extraction and grading); the only change is step 2, where the
reply comes from `ContractAssistant` instead of the raw conversation. The trace keeps exactly what the
user saw, so the simulated user and the grader behave identically across arms.

Call `convcontract.eval.harness.setup()` before importing this module.
"""
from system_agent import SystemAgent
from tasks import get_task
from user_agent import UserAgent
from utils import date_str

from convcontract.layer import ContractAssistant, UserOnlyAssistant


class ConversationSimulatorContract:
    """`policy="user-only"` swaps in the no-extractor ablation (`UserOnlyAssistant`)."""

    def __init__(self, sample, assistant_model, system_model, user_model, extractor_model=None, policy="answer", assistant_temperature=1.0, user_temperature=1.0):
        self.task_name = sample["task"]
        self.task = get_task(self.task_name)
        self.sample = sample
        self.user_agent = UserAgent(self.task, user_model)
        self.system_agent = SystemAgent(self.task_name, system_model, self.sample)
        self.system_message = self.task.generate_system_prompt(self.sample)
        self.user_temperature = user_temperature
        if policy == "user-only":
            self.assistant = UserOnlyAssistant(system_prompt=self.system_message, executor_model=assistant_model, executor_temperature=assistant_temperature)
        else:
            self.assistant = ContractAssistant(
                system_prompt=self.system_message,
                executor_model=assistant_model,
                extractor_model=extractor_model or assistant_model,
                executor_temperature=assistant_temperature,
                policy=policy,
            )
        self.trace = [{"role": "system", "content": self.system_message, "timestamp": date_str()}]

    def run(self, verbose=False):
        is_completed, is_correct, score = False, False, None
        shards = self.sample["shards"]

        while not is_completed:
            revealed_shard_ids = set([msg["content"]["shard_id"] for msg in self.trace if msg["role"] == "log" and msg["content"]["type"] == "shard_revealed"])
            if len(revealed_shard_ids) == len(shards):
                break
            is_last_turn = len(revealed_shard_ids) == len(shards) - 1

            # 1. simulated user turn (unchanged)
            user_response, shard_revealed_id, cost_usd = self.user_agent.generate_response(self.trace, self.sample, temperature=self.user_temperature)
            self.trace.append({"role": "user", "content": user_response, "timestamp": date_str(), "cost_usd": cost_usd})
            if shard_revealed_id != -1:
                self.trace.append({"role": "log", "content": {"type": "shard_revealed", "shard_id": shard_revealed_id}, "timestamp": date_str()})
            if verbose:
                print(f"[user] {user_response}")

            # 2. assistant turn: contract layer instead of the raw conversation
            turn = self.assistant.respond(user_response)
            self.trace.append({"role": "assistant", "content": turn.text, "timestamp": date_str(), "cost_usd": turn.cost_usd})
            contract = {k: v for k, v in turn.contract.items() if k != "log"}
            self.trace.append({"role": "log", "content": {"type": "contract", "decision": turn.decision, "ops": turn.ops, "warnings": turn.warnings, "contract": contract, "executor_prompt": turn.executor_prompt}, "timestamp": date_str()})
            if verbose:
                print(f"[contract] {turn.decision}: {turn.contract}")
                print(f"[assistant] {turn.text}")

            # 3-4. classify the reply, extract and grade answer attempts (unchanged)
            verification, verification_cost = self.system_agent.verify_system_response(self.trace)
            self.trace.append({"role": "log", "content": {"type": "system-verification", "response": verification}, "timestamp": date_str(), "cost_usd": verification_cost})

            if verification["response_type"] == "answer_attempt":
                extracted_answer = self.system_agent.extract_answer(self.trace)
                is_correct, score = None, None
                if self.task_name == "summary" and not is_last_turn:
                    evaluation_return = {"score": 0.0}
                    score = 0.0
                else:
                    evaluation_return = self.task.evaluator_function(extracted_answer, self.sample)
                    is_correct = evaluation_return.get("is_correct", None)
                    score = evaluation_return.get("score", None)
                if score == 1.0 and not is_correct:
                    is_correct = True
                self.trace.append({"role": "log", "content": {"type": "answer-evaluation", "exact_answer": extracted_answer, "is_correct": is_correct, "score": score, "evaluation_return": evaluation_return}, "timestamp": date_str()})
                if verbose:
                    print(f"[eval] score={score}")
                if is_correct:
                    is_completed = True
                    self.trace.append({"role": "log", "content": {"type": "conversation-completed", "is_correct": is_correct}, "timestamp": date_str()})

        return is_correct, score
