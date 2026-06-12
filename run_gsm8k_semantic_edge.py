import asyncio
import json

from mas.datasets.gsm8k_dataset import gsm_data_process, gsm_get_predict
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
    with open("data/gsm8k/gsm8k.jsonl", "r", encoding="utf-8") as f:
        raw_records = [json.loads(line) for line in f]
    records = gsm_data_process(raw_records)
    return shuffled_eval_split(records, args)


def _is_correct(predicted, target, raw_answer, record):
    try:
        return float(predicted) == float(target)
    except (TypeError, ValueError):
        return False


SPEC = SemanticEdgeDatasetSpec(
    dataset="gsm8k",
    split="test",
    cache_name="gsm8k",
    result_prefix="gsm8k_semantic_edge",
    default_domain="gsm8k",
    default_decision_method="FinalRefer",
    default_batch_size=32,
    default_train_set_size=50,
    default_graph_dir="cache/gsm8k/graphs",
    default_role_dir="cache/gsm8k/roles",
    default_model_dir="cache/gsm8k/semantic_edge_models",
    load_eval_records=_load_eval_records,
    make_input=lambda record: {"task": record["task"]},
    task_text=lambda record: record["task"],
    target_answer=lambda record: record["answer"],
    predict_answer=lambda raw_answer, record: gsm_get_predict(_answer_text(raw_answer)),
    is_correct=_is_correct,
    question_text=lambda record: record["task"],
)


if __name__ == "__main__":
    args = parse_semantic_edge_args(
        "Run GSM8K with supervised topology ordering and semantic edge diffusion.",
        SPEC,
    )
    asyncio.run(run_semantic_edge_pipeline(args, SPEC))
