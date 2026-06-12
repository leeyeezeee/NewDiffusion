import asyncio
import json

from mas.datasets.aqua_dataset import aqua_data_process, aqua_get_predict
from semantic_edge_runner import (
    SemanticEdgeDatasetSpec,
    parse_semantic_edge_args,
    run_semantic_edge_pipeline,
    shuffled_eval_split,
)


def _answer_text(raw_answer):
    if isinstance(raw_answer, list):
        return raw_answer[0] if raw_answer else ""
    return raw_answer if isinstance(raw_answer, str) else str(raw_answer)


def _load_eval_records(args):
    with open("data/AQuA/AQuA.jsonl", "r", encoding="utf-8") as f:
        raw_records = [json.loads(line) for line in f]
    records = aqua_data_process(raw_records)
    return shuffled_eval_split(records, args)


SPEC = SemanticEdgeDatasetSpec(
    dataset="aqua",
    split="test",
    cache_name="AQuA",
    result_prefix="aqua_semantic_edge",
    default_domain="aqua",
    default_decision_method="FinalRefer",
    default_batch_size=32,
    default_train_set_size=50,
    default_graph_dir="cache/AQuA/graphs",
    default_role_dir="cache/AQuA/roles",
    default_model_dir="cache/AQuA/semantic_edge_models",
    load_eval_records=_load_eval_records,
    make_input=lambda record: {"task": record["task"]},
    task_text=lambda record: record["task"],
    target_answer=lambda record: record["answer"],
    predict_answer=lambda raw_answer, record: aqua_get_predict(_answer_text(raw_answer)),
    is_correct=lambda predicted, target, raw_answer, record: predicted == target,
    question_text=lambda record: record["task"],
)


if __name__ == "__main__":
    args = parse_semantic_edge_args(
        "Run AQuA with supervised topology ordering and semantic edge diffusion.",
        SPEC,
    )
    asyncio.run(run_semantic_edge_pipeline(args, SPEC))
