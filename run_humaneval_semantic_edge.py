import asyncio
import json

from semantic_edge_runner import (
    SemanticEdgeDatasetSpec,
    parse_semantic_edge_args,
    run_semantic_edge_pipeline,
    shuffled_eval_split,
)


EXECUTOR = None


def _answer_text(raw_answer):
    if isinstance(raw_answer, list):
        return raw_answer[0] if raw_answer else ""
    return raw_answer if isinstance(raw_answer, str) else str(raw_answer)


def _load_eval_records(args):
    with open("data/humaneval/humaneval-py.jsonl", "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    return shuffled_eval_split(records, args)


def _predict_answer(raw_answer, record):
    return _answer_text(raw_answer).lstrip("```python\n").rstrip("\n```")


def _is_correct(predicted, target, raw_answer, record):
    global EXECUTOR
    if EXECUTOR is None:
        from mas.tools.coding.python_executor import PyExecutor
        EXECUTOR = PyExecutor()
    is_solved, _, _ = EXECUTOR.execute(predicted, [record["test"]], timeout=10)
    return bool(is_solved)


SPEC = SemanticEdgeDatasetSpec(
    dataset="humaneval",
    split="test",
    cache_name="humaneval",
    result_prefix="humaneval_semantic_edge",
    default_domain="humaneval",
    default_decision_method="FinalWriteCode",
    default_batch_size=32,
    default_train_set_size=50,
    default_graph_dir="cache/humaneval/graphs",
    default_role_dir="cache/humaneval/roles",
    default_model_dir="cache/humaneval/semantic_edge_models",
    load_eval_records=_load_eval_records,
    make_input=lambda record: {"task": record["prompt"]},
    task_text=lambda record: record["prompt"],
    target_answer=lambda record: record["test"],
    predict_answer=_predict_answer,
    is_correct=_is_correct,
    question_text=lambda record: record["prompt"],
)


if __name__ == "__main__":
    args = parse_semantic_edge_args(
        "Run HumanEval with supervised topology ordering and semantic edge diffusion.",
        SPEC,
    )
    asyncio.run(run_semantic_edge_pipeline(args, SPEC))
