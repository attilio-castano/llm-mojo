"""Select the explicit decode or prefill contract for shared trace tools."""
from . import attention_decode_contract as decode
from . import attention_prefill_contract as prefill

OPERATIONS = (decode.OPERATION,prefill.OPERATION)
ENTRYPOINTS = {**decode.ENTRYPOINTS,**prefill.ENTRYPOINTS}


def target_fields(operation):
    return prefill.TARGET_FIELDS if operation==prefill.OPERATION else decode.TARGET_FIELDS


def configuration(data):
    if data.get('operation')==prefill.OPERATION:
        return prefill.configuration(data)
    return decode.configuration(data)
