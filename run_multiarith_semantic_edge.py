import asyncio
import json

from mas.datasets.multiarith_dataset import multiarith_data_process, multiarith_get_predict
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
    with open("data/MultiArith/MultiArith.json", "r", encoding="utf-8") as f:
        raw_records = json.load(f)
    records = multiarith_data_process(raw_records)
    return shuffled_eval_split(records, args)


def _is_correct(predicted, target, raw_answer, record):
    try:
        return float(predicted) == float(target)
    except (TypeError, ValueError):
        return False


SPEC = SemanticEdgeDatasetSpec(
    dataset="multiarith",
    split="test",
    cache_name="MultiArith",
    result_prefix="multiarith_semantic_edge",
    default_domain="gsm8k",
    default_decision_method="FinalRefer",
    default_batch_size=4,
    default_train_set_size=10,
    default_graph_dir="cache/MultiArith/graphs",
    default_role_dir="cache/MultiArith/roles",
    default_model_dir="cache/MultiArith/semantic_edge_models",
    load_eval_records=_load_eval_records,
    make_input=lambda record: {"task": record["task"]},
    task_text=lambda record: record["task"],
    target_answer=lambda record: record["answer"],
    predict_answer=lambda raw_answer, record: multiarith_get_predict(_answer_text(raw_answer)),
    is_correct=_is_correct,
    question_text=lambda record: record["task"],
)


if __name__ == "__main__":
    args = parse_semantic_edge_args(
        "Run MultiArith with supervised topology ordering and semantic edge diffusion.",
        SPEC,
    )
    asyncio.run(run_semantic_edge_pipeline(args, SPEC))
