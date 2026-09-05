"""Prefill profile identity, including the rectangular causal workload."""
VARIANTS = {0:(0,0,0),1:(0,0,0),2:(4,32,1),3:(8,32,1),4:(16,32,1),
            5:(32,32,1),6:(16,64,1),7:(16,32,1),8:(32,32,1),9:(16,32,2),10:(8,32,4)}
OPERATION = 'grouped_query_attention_prefill'
ENTRYPOINTS = {f'gqa_prefill_{v}': ('enqueue_grouped_query_attention_apple_gpu' if v==0 else
               'enqueue_grouped_query_attention_prefill_materialized_apple_gpu' if v==1 else
               'enqueue_grouped_query_attention_prefill_apple_gpu') for v in VARIANTS}
TARGET_FIELDS = ('profile_workload','dispatches_per_iteration','key_value_rows',
                 'query_heads','key_value_heads','query_tile','key_tile','heads')


def specification(variant,query_rows,key_rows):
    bq,bk,heads = VARIANTS[variant]
    return dict(profile_rows=query_rows,hidden_size=64,key_value_rows=key_rows,
                query_heads=14,key_value_heads=2,query_tile=bq,key_tile=bk,heads=heads,
                profile_workload=f'prefill-r{query_rows}-t{key_rows}-v{variant}',
                dispatches_per_iteration=3 if variant<=1 else 1)


def configuration(data):
    implementation = data.get('implementation','')
    if implementation not in ENTRYPOINTS or data.get('entrypoint') != ENTRYPOINTS[implementation]:
        raise ValueError('invalid prefill implementation identity')
    variant = int(implementation.removeprefix('gqa_prefill_'))
    r,t = data.get('profile_rows'),data.get('key_value_rows')
    if type(r) is not int or type(t) is not int or not 1<=r<=t<=4096:
        raise ValueError('invalid prefill query/key length')
    expected=specification(variant,r,t)
    if any(data.get(k)!=v or (type(v) is int and type(data.get(k)) is not int) for k,v in expected.items()):
        raise ValueError('prefill shape, tile, or dispatch identity mismatch')
    iterations,warmup=data.get('profile_iterations'),data.get('profile_warmup_iterations')
    if type(iterations) is not int or not 1<=iterations*expected['dispatches_per_iteration']<=5000:
        raise ValueError('prefill profile exceeds dispatch budget')
    if type(warmup) is not int or not 0<=warmup<=100:
        raise ValueError('prefill profile warmup is outside bounds')
    return expected
