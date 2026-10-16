"""Plain tensor result interchange; never performs pickle deserialization."""

from types import SimpleNamespace


def normalized_result(data):
    def plain(value):
        if isinstance(value, SimpleNamespace):
            value = vars(value)
        if isinstance(value, dict):
            return {key: plain(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(item) for item in value]
        return value

    return {"format": "dd-normalized-pt-v1", "payload": plain(data)}


def restore_normalized_result(data):
    if not isinstance(data, dict) or data.get("format") != "dd-normalized-pt-v1":
        return data
    payload = data["payload"]
    samples = payload if isinstance(payload, list) else [payload]
    for sample in samples:
        if isinstance(sample, dict) and isinstance(sample.get("data"), dict):
            sample["data"] = SimpleNamespace(**sample["data"])
    return payload
