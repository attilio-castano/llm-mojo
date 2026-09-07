"""Select the explicit decode or prefill contract for shared trace tools."""
from . import attention_decode_contract as decode
from . import attention_prefill_contract as prefill
from . import attention_sublayer_contract as sublayer
from . import mlp_contract as mlp

OPERATIONS = (decode.OPERATION,prefill.OPERATION,sublayer.OPERATION,mlp.OPERATION)
ENTRYPOINTS = {**decode.ENTRYPOINTS,**prefill.ENTRYPOINTS,**sublayer.ENTRYPOINTS,**mlp.ENTRYPOINTS}


def target_fields(operation):
    if operation == mlp.OPERATION:
        return mlp.TARGET_FIELDS
    if operation == sublayer.OPERATION:
        return sublayer.TARGET_FIELDS
    return prefill.TARGET_FIELDS if operation==prefill.OPERATION else decode.TARGET_FIELDS


def configuration(data):
    if data.get('operation') == mlp.OPERATION:
        return mlp.configuration(data)
    if data.get('operation') == sublayer.OPERATION:
        return sublayer.configuration(data)
    if data.get('operation')==prefill.OPERATION:
        return prefill.configuration(data)
    return decode.configuration(data)
