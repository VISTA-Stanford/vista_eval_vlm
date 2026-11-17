import json
import math


def serialize_logprobs(logprobs):
    """
    Serialize vLLM SampleLogprobs (list[dict[int, Logprob]]) to JSON string.
    Each position contains token_id -> {logprob, rank, decoded_token}.
    """
    if logprobs is None:
        return None
    try:
        result = []
        for pos_data in logprobs:
            if pos_data is None:
                result.append(None)
                continue
            pos_list = []
            for token_id, lp_obj in pos_data.items():
                entry = {"token_id": token_id}
                if hasattr(lp_obj, "logprob"):
                    entry["logprob"] = float(lp_obj.logprob)
                if hasattr(lp_obj, "rank") and lp_obj.rank is not None:
                    entry["rank"] = int(lp_obj.rank)
                if hasattr(lp_obj, "decoded_token") and lp_obj.decoded_token is not None:
                    entry["decoded_token"] = str(lp_obj.decoded_token)
                pos_list.append(entry)
            result.append(pos_list)
        return json.dumps(result)
    except Exception:
        return None


#: Passed to an API adapter in place of an options list: force JSON {"probability": whole number 0-100} instead of
#: a choice (eval-set recipes with ``answer_form: "probability"``). The schema says integer; the 0-100 range is
#: checked on the reply (``read_probability``), not in the schema, because Claude's and OpenAI's structured output do
#: not all honour minimum/maximum.
PROBABILITY = "probability_0_100"
PROBABILITY_DESCRIPTION = "Probability, from 0 to 100, that the answer is yes. One whole number."


def read_probability(text):
    """(JSON text {"probability": n}, error) from a forced reply. A number that is not whole, or outside 0-100, is
    an error, not a guess (so it is not cached and is asked again)."""
    try:
        v = json.loads(text)
        p = v.get("probability")
    except (json.JSONDecodeError, AttributeError, TypeError):
        return None, f"answer is not the schema's JSON: {str(text)[:100]!r}"
    if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or p != int(p):
        return None, f"probability {p!r} is not a whole number"
    if not 0 <= p <= 100:
        return None, f"probability {p!r} is outside 0-100"
    return json.dumps({"probability": int(p)}), None


class BaseVLMAdapter:
    # Capability seam for the context assembler (Phase 1). Models that support
    # mid-prompt image interleaving (assembly `inline_by_timestamp`) flip this to
    # True in Phase 1.5; until then every model is fail-closed and the assembler
    # raises at preflight — before weights load — if a config requests inline.
    supports_inline: bool = False

    # Text-token budget denominator for the Phase-2 config-context viewer. A CLASS
    # attribute (not set in `load()`) so the WEIGHT-FREE viewer can read it off the
    # constructed adapter without a GPU / weights load. Each subclass sets it to the
    # `max_model_len` (or InternVL's `target_seq_len`) it passes at load; `None` here
    # means "context window unknown" -> the viewer renders no false denominator.
    # (Kept as a mirrored literal rather than rewiring load() to read it, to avoid
    # touching the weight-bearing inference path — see the plan's adapter table.)
    context_window: int | None = None

    def __init__(self, model_name, device="auto", cache_dir=None):
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir

    # def create_template(self, item):
    #     conversation = {
    #             "role": "user",
    #             "content": [
    #                 {"type": "image", "image": item["image"]},
    #                 {"type": "text", "text": item["question"]},
    #             ],
    #         }
    #     return conversation

    def create_template(self, item):
        return [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are a helpful assistant."}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": item["image"]},
                    {"type": "text", "text": item["question"]},
                ],
            }
        ]

    def prepare_inputs(self, messages, processor, model):
        raise NotImplementedError

    def infer(self, model, processor, inputs, max_new_tokens):
        raise NotImplementedError