import asyncio
import json

from mas.datasets.svamp_dataset import svamp_data_process, svamp_get_predict
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
    with open("data/SVAMP/SVAMP.json", "r", encoding="utf-8") as f:
        raw_records = json.load(f)
    records = svamp_data_process(raw_records)
    return shuffled_eval_split(records, args)


def _is_correct(predicted, target, raw_answer, record):
    try:
        return float(predicted) == float(target)
    except (TypeError, ValueError):
        return False


SPEC = SemanticEdgeDatasetSpec(
    dataset="svamp",
    split="test",
    cache_name="SVAMP",
    result_prefix="svamp_semantic_edge",
    default_domain="gsm8k",
    default_decision_method="FinalRefer",
    default_batch_size=32,
    default_train_set_size=50,
    default_graph_dir="cache/SVAMP/graphs",
    default_role_dir="cache/SVAMP/roles",
    default_model_dir="cache/SVAMP/semantic_edge_models",
    load_eval_records=_load_eval_records,
    make_input=lambda record: {"task": record["task"]},
    task_text=lambda record: record["task"],
    target_answer=lambda record: record["answer"],
    predict_answer=lambda raw_answer, record: svamp_get_predict(_answer_text(raw_answer)),
    is_correct=_is_correct,
    question_text=lambda record: record["task"],
)


if __name__ == "__main__":
    args = parse_semantic_edge_args(
        "Run SVAMP with supervised topology ordering and semantic edge diffusion.",
        SPEC,
    )
    asyncio.run(run_semantic_edge_pipeline(args, SPEC))
