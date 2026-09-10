"""Composed decoder boundaries and cache state, against the pinned CPU layer."""
from layout import TensorLayout, TileTensor, row_major, col_major
from llm_mojo.rms_norm import enqueue_rms_norm_apple_gpu
from llm_mojo.rope import enqueue_rope_apple_gpu
from llm_mojo.residual import enqueue_residual_apple_gpu
from llm_mojo.attention import enqueue_grouped_query_attention_apple_gpu
from llm_mojo.attention_decode import enqueue_grouped_query_attention_decode_apple_gpu, enqueue_grouped_query_attention_consistent_apple_gpu
from llm_mojo.attention_prefill import enqueue_grouped_query_attention_prefill_apple_gpu, enqueue_grouped_query_attention_prefill_split_apple_gpu
from max.gpu.host import DeviceContext, DeviceBuffer
from std.testing import TestSuite, assert_equal, assert_raises
from std.python import PythonObject
from llm_mojo.attention_sublayer import AttentionWeights, AttentionCache, AttentionWorkspace, _enqueue_attention_qkv, _enqueue_attention_wo
from llm_mojo.mlp import MLPWeights, MLPWorkspace, enqueue_mlp_stage_apple_gpu, enqueue_mlp_apple_gpu
from llm_mojo.decoder_layer import enqueue_decoder_layer, decoder_mappings, decoder_policy_configuration, _enqueue_decoder_layer_policy_storage
from decoder_layer_support import decoder_support, load_decoder, check_decoder, poison_decoder, decoder_snapshot, exact_decoder, check_decoder_cache, check_decoder_active, check_decoder_slice



def _case_mappings(policy: Int, rows: Int, total_rows: Int) raises -> SIMD[DType.int64,4]:
    var variant = policy
    if policy >= 100 and policy <= 103:
        variant = decoder_policy_configuration(policy%2 == 1, rows, total_rows,
            24 if policy >= 102 else 1)
    return decoder_mappings(variant,rows)


def _enqueue_policy_case[XL: TensorLayout](
    ctx: DeviceContext, mut aw: AttentionWeights, mut cache: AttentionCache,
    mut a: AttentionWorkspace, mut mw: MLPWeights, mut m: MLPWorkspace,
    x: TileTensor[DType.bfloat16,XL,MutAnyOrigin], mapping: Int,
    integrated: Bool, gqa: Int, projection: Int, policy: Int,
) raises -> Int:
    if policy == 100 or policy == 102:
        return _enqueue_decoder_layer_policy_storage[False](ctx,aw,cache,a,mw,m,x,
            24 if policy == 102 else 1)
    if policy == 101 or policy == 103:
        return _enqueue_decoder_layer_policy_storage[True](ctx,aw,cache,a,mw,m,x,
            24 if policy == 103 else 1)
    return enqueue_decoder_layer(ctx,aw,cache,a,mw,m,x,mapping,integrated,gqa,projection)


def test_tiny_decoder_reference() raises:
    var name = String("h8_i12_nq2_nk1_d4_t7_s4001_base")
    var support = decoder_support()
    support.verify_case(name)
    support.configure(name, 0, "full", "layer")
    var ctx = DeviceContext()
    assert_equal(ctx.api(), "metal")
    print("decoder device", ctx.name(), "backend", ctx.api())
    var aw = AttentionWeights(ctx, 2, 1, 4)
    var mw = MLPWeights(ctx, 8, 12)
    var cache = AttentionCache(ctx, 7, 1, 4)
    var attention = AttentionWorkspace(ctx, 7, 7, 2, 1, 4)
    var mlp = MLPWorkspace(ctx, 7, 8, 12)
    var x = ctx.enqueue_create_buffer[DType.bfloat16](7 * 8)
    load_decoder(x, name, "input_X", 0, 7 * 8)
    load_decoder(aw.norm, name, "input_input_norm", 0, 8)
    load_decoder(aw.qkv, name, "input_qkv", 0, 16 * 8)
    load_decoder(aw.bias, name, "input_bias", 0, 16)
    load_decoder(aw.output, name, "input_wo", 0, 8 * 8)
    load_decoder(mw.norm, name, "input_post_norm", 0, 8)
    load_decoder(mw.gate, name, "input_gate", 0, 12 * 8)
    load_decoder(mw.up, name, "input_up", 0, 12 * 8)
    load_decoder(mw.down, name, "input_down", 0, 8 * 12)
    load_decoder(attention.cosine, name, "full_cosine", 0, 7 * 4)
    load_decoder(attention.sine, name, "full_sine", 0, 7 * 4)
    var route = enqueue_decoder_layer(ctx, aw, cache, attention, mw, mlp,
                                      TileTensor(x, row_major(7, 8)), integrated=False)
    ctx.synchronize()
    assert_equal(route, 3)
    assert_equal(cache.length, 7)
    check_decoder(attention.projected, name, "B_att", "full", 0, 7, 8)
    check_decoder(attention.output, name, "Z", "full", 0, 7, 8)
    check_decoder(mlp.down, name, "B_mlp", "full", 0, 7, 8)
    check_decoder(mlp.output, name, "Y", "full", 0, 7, 8)



def _load_weights(mut aw: AttentionWeights, mut mw: MLPWeights,
                  mut a: AttentionWorkspace, name: String, t: Int) raises:
    var h = aw.hidden
    var k = aw.kv_heads * aw.head_dim
    var i = mw.intermediate
    load_decoder(aw.norm, name, "input_input_norm", 0, h)
    load_decoder(aw.qkv, name, "input_qkv", 0, (h + 2*k)*h)
    load_decoder(aw.bias, name, "input_bias", 0, h + 2*k)
    load_decoder(aw.output, name, "input_wo", 0, h*h)
    load_decoder(mw.norm, name, "input_post_norm", 0, h)
    load_decoder(mw.gate, name, "input_gate", 0, i*h)
    load_decoder(mw.up, name, "input_up", 0, i*h)
    load_decoder(mw.down, name, "input_down", 0, h*i)
    poison_decoder(a.cosine, 0)
    poison_decoder(a.sine, 0)
    load_decoder(a.cosine, name, "full_cosine", 0, t*aw.head_dim)
    load_decoder(a.sine, name, "full_sine", 0, t*aw.head_dim)


def _poison(mut a: AttentionWorkspace, mut m: MLPWorkspace, r: Int) raises:
    var h = a.query_heads * a.head_dim
    var k = a.kv_heads * a.head_dim
    var i = m.intermediate
    poison_decoder(a.normalized, r*h)
    poison_decoder(a.raw_query, r*h)
    poison_decoder(a.raw_key, r*k)
    poison_decoder(a.raw_value, r*k)
    poison_decoder(a.query, r*h)
    poison_decoder(a.rotated_key, r*k)
    poison_decoder(a.attention, r*h)
    poison_decoder(a.projected, r*h)
    poison_decoder(a.output, r*h)
    poison_decoder(a.packed, r*(h+2*k))
    poison_decoder(m.normalized, r*h)
    poison_decoder(m.gate, r*i)
    poison_decoder(m.up, r*i)
    poison_decoder(m.activated, r*i)
    poison_decoder(m.gated, r*i)
    poison_decoder(m.down, r*h)
    poison_decoder(m.output, r*h)
    a.fp32_scratch.enqueue_fill(Float32(FloatLiteral.nan))
    a.split.enqueue_fill(Float32(FloatLiteral.nan))
    a.prefill_partial.enqueue_fill(Float32(FloatLiteral.nan))


def _check_layer(mut a: AttentionWorkspace, mut m: MLPWorkspace, name: String,
                 schedule: String, p: Int, r: Int) raises:
    var h = a.query_heads*a.head_dim
    var k = a.kv_heads*a.head_dim
    var i = m.intermediate
    check_decoder(a.normalized, name, "N_att", schedule, p, r, h)
    check_decoder(a.raw_query, name, "Q_raw", schedule, p, r, h)
    check_decoder(a.raw_key, name, "K_raw", schedule, p, r, k)
    check_decoder(a.raw_value, name, "V_raw", schedule, p, r, k)
    check_decoder(a.query, name, "Q", schedule, p, r, h)
    check_decoder(a.rotated_key, name, "K_rot", schedule, p, r, k)
    check_decoder(a.attention, name, "O", schedule, p, r, h)
    check_decoder(a.projected, name, "B_att", schedule, p, r, h)
    check_decoder(a.output, name, "Z", schedule, p, r, h)
    check_decoder(m.normalized, name, "N_mlp", schedule, p, r, h)
    check_decoder(m.gate, name, "G", schedule, p, r, i)
    check_decoder(m.up, name, "U", schedule, p, r, i)
    check_decoder(m.activated, name, "A", schedule, p, r, i)
    check_decoder(m.gated, name, "S", schedule, p, r, i)
    check_decoder(m.down, name, "B_mlp", schedule, p, r, h)
    check_decoder(m.output, name, "Y", schedule, p, r, h)


def _schedule(ctx: DeviceContext, mut aw: AttentionWeights, mut cache: AttentionCache,
              mut a: AttentionWorkspace, mut mw: MLPWeights, mut m: MLPWorkspace,
              mut xb: DeviceBuffer[DType.bfloat16], name: String, schedule: String,
              calls: PythonObject, policy: Int, selection: Bool) raises:
    var support = decoder_support()
    var nq = aw.query_heads
    var nk = aw.kv_heads
    var d = aw.head_dim
    var h = aw.hidden
    support.configure(name, policy, schedule, "layer")
    cache.reset(ctx)
    poison_decoder(cache.key, 0)
    poison_decoder(cache.value, 0)
    for c in range(Int(py=calls.__len__())):
        var p = Int(py=calls[c][0])
        var r = Int(py=calls[c][1])
        var gqa = Int(_case_mappings(policy,r,p+r)[0]) if selection else 0
        var projection = Int(_case_mappings(policy,r,p+r)[1]) if selection else 0
        var mapping = Int(_case_mappings(policy,r,p+r)[2]) if selection else (policy if r > 1 else 0)
        assert_equal(cache.length, p)
        _poison(a, m, r)
        var prefix_k = decoder_snapshot(cache.key, p*nk*d)
        var prefix_v = decoder_snapshot(cache.value, p*nk*d)
        var route = _enqueue_policy_case(ctx, aw, cache, a, mw, m,
            TileTensor(xb.unsafe_ptr().unsafe_offset(p*h), row_major(r,h)),
            mapping, nq == 14, gqa, projection, policy if selection else -1)
        assert_equal(route, (11 if gqa == 5 else (4 if r == 1 else 6+gqa)) if nq == 14 else 3)
        assert_equal(cache.length, p+r)
        ctx.synchronize()
        support.route(route, mapping, p, r, ctx.name(), ctx.api())
        _check_layer(a, m, name, schedule, p, r)
        var label = "full_" if schedule == "full" else schedule+"_"+String(p)+"_"
        check_decoder_cache(cache.key, a.rotated_key, prefix_k, name,
                            label+"cache_key", p, r, nk*d, cache.capacity)
        check_decoder_cache(cache.value, a.raw_value, prefix_v, name,
                            label+"cache_value", p, r, nk*d, cache.capacity)


def _case(name: String, t: Int, h: Int, nq: Int, nk: Int, d: Int, i: Int,
          policy: Int, selection: Bool = False) raises:
    var support = decoder_support()
    support.verify_case(name)
    var ctx = DeviceContext()
    assert_equal(ctx.api(), "metal")
    print("decoder case", name, "policy", policy, "device", ctx.name(), ctx.api())
    var cap = t + 1 if t < 4096 else t
    var aw = AttentionWeights(ctx, nq, nk, d)
    var mw = MLPWeights(ctx, h, i)
    var gqa = Int(_case_mappings(policy,t,t)[0]) if selection else 0
    var projection = Int(_case_mappings(policy,t,t)[1]) if selection else 0
    var splits = 8 if gqa == 4 or policy == 100 or policy == 102 else 1
    var a = AttentionWorkspace(ctx, cap, cap, nq, nk, d, False, nq != 14, splits)
    var m = MLPWorkspace(ctx, cap, h, i)
    var cache = AttentionCache(ctx, cap, nk, d)
    var xb = ctx.enqueue_create_buffer[DType.bfloat16](t*h)
    load_decoder(xb, name, "input_X", 0, t*h)
    _load_weights(aw, mw, a, name, t)
    var saved_xb = decoder_snapshot(xb, len(xb))
    var saved_aw_norm = decoder_snapshot(aw.norm, len(aw.norm))
    var saved_aw_qkv = decoder_snapshot(aw.qkv, len(aw.qkv))
    var saved_aw_bias = decoder_snapshot(aw.bias, len(aw.bias))
    var saved_aw_output = decoder_snapshot(aw.output, len(aw.output))
    var saved_mw_norm = decoder_snapshot(mw.norm, len(mw.norm))
    var saved_mw_gate = decoder_snapshot(mw.gate, len(mw.gate))
    var saved_mw_up = decoder_snapshot(mw.up, len(mw.up))
    var saved_mw_down = decoder_snapshot(mw.down, len(mw.down))
    var saved_a_cosine = decoder_snapshot(a.cosine, len(a.cosine))
    var saved_a_sine = decoder_snapshot(a.sine, len(a.sine))
    var schedules = support.schedules(name)
    for s in range(Int(py=schedules.__len__())):
        var schedule = String(py=schedules[s][0])
        var calls = schedules[s][1]
        if schedule == "policy_tokenwise":
            # Decode owns one active row and one fully checked guard row.
            # Keep the original full-sized workspace alive and protected too.
            var da = AttentionWorkspace(ctx, 2, cap, nq, nk, d, False, nq != 14, splits)
            var dm = MLPWorkspace(ctx, 2, h, i)
            poison_decoder(da.cosine, 0)
            poison_decoder(da.sine, 0)
            load_decoder(da.cosine, name, "full_cosine", 0, t*d)
            load_decoder(da.sine, name, "full_sine", 0, t*d)
            var dc = decoder_snapshot(da.cosine, len(da.cosine))
            var ds = decoder_snapshot(da.sine, len(da.sine))
            _schedule(ctx,aw,cache,da,mw,dm,xb,name,schedule,calls,policy,selection)
            exact_decoder(da.cosine, dc, "tokenwise_cosine")
            exact_decoder(da.sine, ds, "tokenwise_sine")
        else:
            _schedule(ctx,aw,cache,a,mw,m,xb,name,schedule,calls,policy,selection)
    exact_decoder(xb, saved_xb, "xb")
    exact_decoder(aw.norm, saved_aw_norm, "aw_norm")
    exact_decoder(aw.qkv, saved_aw_qkv, "aw_qkv")
    exact_decoder(aw.bias, saved_aw_bias, "aw_bias")
    exact_decoder(aw.output, saved_aw_output, "aw_output")
    exact_decoder(mw.norm, saved_mw_norm, "mw_norm")
    exact_decoder(mw.gate, saved_mw_gate, "mw_gate")
    exact_decoder(mw.up, saved_mw_up, "mw_up")
    exact_decoder(mw.down, saved_mw_down, "mw_down")
    exact_decoder(a.cosine, saved_a_cosine, "a_cosine")
    exact_decoder(a.sine, saved_a_sine, "a_sine")
    _local_attention(ctx, aw, a, name, t, policy, gqa, projection)
    _local_mlp(ctx, mw, m, name, t,
               Int(_case_mappings(policy,t,t)[2]) if selection else policy, policy)


def test_decoder_development() raises:
    var support = decoder_support()
    var cases = support.cases()
    for j in range(Int(py=cases.__len__())):
        var spec = cases[j]
        var name = String(py=spec[0])
        for policy in [0,7]:
            if policy == 7 and (Int(py=spec[1]) == 1 or Int(py=spec[3]) != 14):
                continue
            _case(name, Int(py=spec[1]), Int(py=spec[2]), Int(py=spec[3]),
                  Int(py=spec[4]), Int(py=spec[5]), Int(py=spec[6]), policy)
    support.result_summary()


def _local_mlp(ctx: DeviceContext, mut w: MLPWeights, mut m: MLPWorkspace,
               name: String, r: Int, mapping: Int, record_policy: Int = -1) raises:
    var support = decoder_support()
    var policy = mapping if record_policy < 0 else record_policy
    support.configure(name, policy, "full", "operation")
    var h = w.hidden
    var i = w.intermediate
    var z = ctx.enqueue_create_buffer[DType.bfloat16](r*h)
    load_decoder(z, name, "full_Z", 0, r*h)
    var x = TileTensor(z, row_major(r,h))
    poison_decoder(m.normalized, r*h)
    enqueue_mlp_stage_apple_gpu(ctx,w,m,x,0,mapping)
    check_decoder(m.normalized,name,"N_mlp","full",0,r,h,"operation")
    load_decoder(m.normalized, name, "full_N_mlp", 0, r*h)
    poison_decoder(m.gate, r*i)
    enqueue_mlp_stage_apple_gpu(ctx,w,m,x,1,mapping)
    check_decoder(m.gate,name,"G","full",0,r,i,"operation")
    poison_decoder(m.up, r*i)
    enqueue_mlp_stage_apple_gpu(ctx,w,m,x,2,mapping)
    check_decoder(m.up,name,"U","full",0,r,i,"operation")
    load_decoder(m.gate, name, "full_G", 0, r*i)
    poison_decoder(m.activated, r*i)
    enqueue_mlp_stage_apple_gpu(ctx,w,m,x,3,mapping)
    check_decoder(m.activated,name,"A","full",0,r,i,"operation")
    load_decoder(m.activated, name, "full_A", 0, r*i)
    load_decoder(m.up, name, "full_U", 0, r*i)
    poison_decoder(m.gated, r*i)
    enqueue_mlp_stage_apple_gpu(ctx,w,m,x,4,mapping)
    check_decoder(m.gated,name,"S","full",0,r,i,"operation")
    load_decoder(m.gated, name, "full_S", 0, r*i)
    poison_decoder(m.down, r*h)
    enqueue_mlp_stage_apple_gpu(ctx,w,m,x,5,mapping)
    check_decoder(m.down,name,"B_mlp","full",0,r,h,"operation")
    load_decoder(m.down, name, "full_B_mlp", 0, r*h)
    poison_decoder(m.output, r*h)
    enqueue_mlp_stage_apple_gpu(ctx,w,m,x,6,mapping)
    check_decoder(m.output,name,"Y","full",0,r,h,"operation")
    support.configure(name, policy, "full", "mlp")
    poison_decoder(m.down,r*h)
    poison_decoder(m.output,r*h)
    enqueue_mlp_apple_gpu(ctx,w,m,x,mapping)
    check_decoder(m.down,name,"B_mlp","full",0,r,h,"mlp")
    check_decoder(m.output,name,"Y","full",0,r,h,"mlp")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()


def _local_attention(ctx: DeviceContext, mut w: AttentionWeights,
                     mut a: AttentionWorkspace, name: String, r: Int, policy: Int,
                     gqa: Int = 0, projection: Int = 0) raises:
    var support = decoder_support()
    support.configure(name, policy, "full", "operation")
    var h = w.hidden
    var nk = w.kv_heads
    var nq = w.query_heads
    var d = w.head_dim
    var k = nk*d
    var xb = ctx.enqueue_create_buffer[DType.bfloat16](r*h)
    load_decoder(xb,name,"input_X",0,r*h)
    var x = TileTensor(xb,row_major(r,h))
    poison_decoder(a.normalized,r*h)
    enqueue_rms_norm_apple_gpu(ctx,x,TileTensor(w.norm,row_major(h)),
                               TileTensor(a.normalized,row_major(r,h)))
    check_decoder(a.normalized,name,"N_att","full",0,r,h,"operation")
    load_decoder(a.normalized,name,"full_N_att",0,r*h)
    poison_decoder(a.raw_query,r*h)
    poison_decoder(a.raw_key,r*k)
    poison_decoder(a.raw_value,r*k)
    _enqueue_attention_qkv(ctx,w,a,r,(2 if projection == 6 else projection - 2) if projection >= 6 else
        (((3 if projection == 5 else 2) if r >= 16 else 1) if nq == 14 and gqa != 5 else 0))
    check_decoder(a.raw_query,name,"Q_raw","full",0,r,h,"operation")
    check_decoder(a.raw_key,name,"K_raw","full",0,r,k,"operation")
    check_decoder(a.raw_value,name,"V_raw","full",0,r,k,"operation")
    load_decoder(a.raw_query,name,"full_Q_raw",0,r*h)
    load_decoder(a.raw_key,name,"full_K_raw",0,r*k)
    poison_decoder(a.query,r*h)
    poison_decoder(a.rotated_key,r*k)
    var cosine = TileTensor(a.cosine,row_major(a.capacity,d))
    var sine = TileTensor(a.sine,row_major(a.capacity,d))
    enqueue_rope_apple_gpu(ctx,TileTensor(a.raw_query,row_major(r,nq,d)),cosine,sine,
                           TileTensor(a.query,row_major(r,nq,d)),0)
    enqueue_rope_apple_gpu(ctx,TileTensor(a.raw_key,row_major(r,nk,d)),cosine,sine,
                           TileTensor(a.rotated_key,row_major(r,nk,d)),0)
    check_decoder(a.query,name,"Q","full",0,r,h,"operation")
    check_decoder(a.rotated_key,name,"K_rot","full",0,r,k,"operation")
    load_decoder(a.query,name,"full_Q",0,r*h)
    load_decoder(a.rotated_key,name,"full_K_rot",0,r*k)
    load_decoder(a.raw_value,name,"full_V_raw",0,r*k)
    poison_decoder(a.attention,r*h)
    var q = TileTensor(a.query,row_major(r,nq,d))
    var keys = TileTensor(a.rotated_key,row_major(r,nk,d))
    var values = TileTensor(a.raw_value,row_major(r,nk,d))
    var out = TileTensor(a.attention,row_major(r,nq,d))
    if nq != 14:
        enqueue_grouped_query_attention_apple_gpu(ctx,q,keys,values,
            TileTensor(a.fp32_scratch,row_major(r,nq,r)),out)
    elif gqa == 5:
        enqueue_grouped_query_attention_consistent_apple_gpu(ctx,q,keys,values,out,
            TileTensor(a.split,row_major(14,1,66)))
    elif r == 1:
        enqueue_grouped_query_attention_decode_apple_gpu[32,1,1,fp32_scores=True](
            ctx,q,keys,values,out,TileTensor(a.split,row_major(14,1,66)))
    elif gqa == 4:
        enqueue_grouped_query_attention_prefill_split_apple_gpu[8](ctx,q,keys,values,
            TileTensor(a.prefill_partial,row_major(r,14,8,66)),out)
    else:
        enqueue_grouped_query_attention_prefill_apple_gpu[32,32,MMA=True,SCHEDULE=2,FP32=True](ctx,q,keys,values,out)
    check_decoder(a.attention,name,"O","full",0,r,h,"operation")
    load_decoder(a.attention,name,"full_O",0,r*h)
    poison_decoder(a.projected,r*h)
    _enqueue_attention_wo(ctx,w,a,r,projection >= 6 or (nq == 14 and r >= 16 and gqa != 5),
        projection - 4 if projection >= 7 else (1 if projection == 5 else 0))
    check_decoder(a.projected,name,"B_att","full",0,r,h,"operation")
    load_decoder(a.projected,name,"full_B_att",0,r*h)
    poison_decoder(a.output,r*h)
    enqueue_residual_apple_gpu(ctx,x,TileTensor(a.projected,row_major(r,h)),
                               TileTensor(a.output,row_major(r,h)))
    check_decoder(a.output,name,"Z","full",0,r,h,"operation")


def test_whole_call_preflight() raises:
    var support = decoder_support()
    support.configure("behavior",0,"invalid","preflight")
    var ctx = DeviceContext()
    var aw = AttentionWeights(ctx,2,1,4)
    var mw = MLPWeights(ctx,8,12)
    var a = AttentionWorkspace(ctx,8,8,2,1,4)
    var m = MLPWorkspace(ctx,8,8,12)
    var cache = AttentionCache(ctx,8,1,4)
    var xb = ctx.enqueue_create_buffer[DType.bfloat16](8*8)
    _poison(a,m,0)
    poison_decoder(cache.key,0)
    poison_decoder(cache.value,0)
    var before_k = decoder_snapshot(cache.key,len(cache.key))
    var before_v = decoder_snapshot(cache.value,len(cache.value))
    var before_z = decoder_snapshot(a.output,len(a.output))
    var before_n = decoder_snapshot(a.normalized,len(a.normalized))
    var before_y = decoder_snapshot(m.output,len(m.output))
    var before_mn = decoder_snapshot(m.normalized,len(m.normalized))
    for rows in [0,9]:
        with assert_raises():
            _ = enqueue_decoder_layer(ctx,aw,cache,a,mw,m,TileTensor(xb,row_major(rows,8)),integrated=False)
    with assert_raises():
        _ = enqueue_decoder_layer(ctx,aw,cache,a,mw,m,TileTensor(xb,row_major(1,7)),integrated=False)
    with assert_raises():
        _ = enqueue_decoder_layer(ctx,aw,cache,a,mw,m,TileTensor(xb,col_major(7,8)),integrated=False)
    for mapping in [-1,1,8,99]:
        with assert_raises():
            _ = enqueue_decoder_layer(ctx,aw,cache,a,mw,m,TileTensor(xb,row_major(7,8)),mapping,False)
    with assert_raises():
        _ = enqueue_decoder_layer(ctx,aw,cache,a,mw,m,TileTensor(xb,row_major(1,8)),7,False)
    with assert_raises():
        _ = enqueue_decoder_layer(ctx,aw,cache,a,mw,m,TileTensor(xb,row_major(7,8)),integrated=False,gqa_mapping=1)
    with assert_raises():
        _ = enqueue_decoder_layer(ctx,aw,cache,a,mw,m,TileTensor(xb,row_major(7,8)),integrated=False,projection_mapping=1)
    # Attention alone is valid, but the composed preflight must reject MLP.
    m.hidden = 7
    with assert_raises():
        _ = enqueue_decoder_layer(ctx,aw,cache,a,mw,m,TileTensor(xb,row_major(7,8)),integrated=False)
    m.hidden = 8
    m.max_rows = 1
    with assert_raises():
        _ = enqueue_decoder_layer(ctx,aw,cache,a,mw,m,TileTensor(xb,row_major(7,8)),integrated=False)
    m.max_rows = 8
    a.fp32_materialized = False
    with assert_raises():
        _ = enqueue_decoder_layer(ctx,aw,cache,a,mw,m,TileTensor(xb,row_major(7,8)),integrated=False)
    a.fp32_materialized = True
    cache.length = 8
    with assert_raises():
        _ = enqueue_decoder_layer(ctx,aw,cache,a,mw,m,TileTensor(xb,row_major(1,8)),integrated=False)
    assert_equal(cache.length,8)
    cache.length = 0
    # A pointer-created view cannot conceal overlap with a writable output.
    var aliased_input = m.output.unsafe_ptr()
    with assert_raises(contains="overlap"):
        _ = enqueue_decoder_layer(ctx,aw,cache,a,mw,m,TileTensor(aliased_input,row_major(7,8)),integrated=False)
    m.down = ctx.enqueue_create_buffer[DType.bfloat16](1)
    with assert_raises(contains="smaller"):
        _ = enqueue_decoder_layer(ctx,aw,cache,a,mw,m,TileTensor(xb,row_major(7,8)),integrated=False)
    assert_equal(cache.length,0)
    ctx.synchronize()
    exact_decoder(cache.key,before_k,"invalid cache key")
    exact_decoder(cache.value,before_v,"invalid cache value")
    exact_decoder(a.output,before_z,"invalid attention output")
    exact_decoder(a.normalized,before_n,"invalid first attention dispatch")
    exact_decoder(m.output,before_y,"invalid MLP output")
    exact_decoder(m.normalized,before_mn,"invalid first MLP dispatch")


def test_asynchronous_twelve_decode_calls() raises:
    _asynchronous_decoder()


def _asynchronous_decoder(variant: Int = -1) raises:
    var name = String("h896_i4864_nq14_nk2_d64_t65_s4001_base")
    var support = decoder_support()
    support.verify_case(name)
    support.configure(name,variant if variant >= 0 else 7,"reuse","async")
    var ctx = DeviceContext()
    assert_equal(ctx.api(),"metal")
    var aw = AttentionWeights(ctx)
    var mw = MLPWeights(ctx)
    var gqa = Int(_case_mappings(variant,53,53)[0]) if variant >= 0 else 0
    var projection = Int(_case_mappings(variant,53,53)[1]) if variant >= 0 else 0
    var splits = 8 if gqa == 4 or variant == 100 or variant == 102 else 1
    var a = AttentionWorkspace(ctx,65,66,14,2,64,False,False,splits)
    var m = MLPWorkspace(ctx,65)
    var cache = AttentionCache(ctx,66)
    var xb = ctx.enqueue_create_buffer[DType.bfloat16](65*896)
    load_decoder(xb,name,"input_X",0,65*896)
    _load_weights(aw,mw,a,name,65)
    poison_decoder(cache.key,0)
    poison_decoder(cache.value,0)
    var outputs = List[DeviceBuffer[DType.bfloat16]]()
    var branches = List[DeviceBuffer[DType.bfloat16]]()
    var att_outputs = List[DeviceBuffer[DType.bfloat16]]()
    var att_branches = List[DeviceBuffer[DType.bfloat16]]()
    for _ in range(13):
        outputs.append(ctx.enqueue_create_buffer[DType.bfloat16](65*896))
        branches.append(ctx.enqueue_create_buffer[DType.bfloat16](65*896))
        att_outputs.append(ctx.enqueue_create_buffer[DType.bfloat16](65*896))
        att_branches.append(ctx.enqueue_create_buffer[DType.bfloat16](65*896))
    ctx.synchronize()
    # No host mapping, wait, allocation or Python invocation in this sequence.
    for j in range(13):
        var p = 0 if j == 0 else 52+j
        var r = 53 if j == 0 else 1
        assert_equal(cache.length,p)
        var route = _enqueue_policy_case(ctx,aw,cache,a,mw,m,
            TileTensor(xb.unsafe_ptr().unsafe_offset(p*896),row_major(r,896)),
            Int(_case_mappings(variant,r,p+r)[2]) if variant >= 0 else (7 if r > 1 else 0),
            True,gqa,projection,variant)
        assert_equal(route,11 if gqa == 5 else (6+gqa if r > 1 else 4))
        assert_equal(cache.length,p+r)
        ctx.enqueue_copy(dst_buf=outputs[j],src_buf=m.output)
        ctx.enqueue_copy(dst_buf=branches[j],src_buf=m.down)
        ctx.enqueue_copy(dst_buf=att_outputs[j],src_buf=a.output)
        ctx.enqueue_copy(dst_buf=att_branches[j],src_buf=a.projected)
    ctx.synchronize()
    for j in range(13):
        var p = 0 if j == 0 else 52+j
        var r = 53 if j == 0 else 1
        check_decoder_active(outputs[j],name,"Y","reuse",p,r,896)
        check_decoder_active(branches[j],name,"B_mlp","reuse",p,r,896)
        check_decoder_active(att_outputs[j],name,"Z","reuse",p,r,896)
        check_decoder_active(att_branches[j],name,"B_att","reuse",p,r,896)
    # A second cache and workspace execute the same schedule with explicit waits.
    var separate_a = AttentionWorkspace(ctx,65,66,14,2,64,False,False,splits)
    var separate_m = MLPWorkspace(ctx,65)
    var separate_cache = AttentionCache(ctx,66)
    _load_weights(aw,mw,separate_a,name,65)
    poison_decoder(separate_cache.key,0)
    poison_decoder(separate_cache.value,0)
    for j in range(13):
        var p = 0 if j == 0 else 52+j
        var r = 53 if j == 0 else 1
        _ = _enqueue_policy_case(ctx,aw,separate_cache,separate_a,mw,separate_m,
            TileTensor(xb.unsafe_ptr().unsafe_offset(p*896),row_major(r,896)),
            Int(_case_mappings(variant,r,p+r)[2]) if variant >= 0 else (7 if r > 1 else 0),
            True,gqa,projection,variant)
        ctx.synchronize()
        exact_decoder(separate_m.output,decoder_snapshot(outputs[j],r*896),"async Y vs separate workspace")
        exact_decoder(separate_m.down,decoder_snapshot(branches[j],r*896),"async B_mlp vs separate workspace")
        exact_decoder(separate_a.output,decoder_snapshot(att_outputs[j],r*896),"async Z vs separate workspace")
        exact_decoder(separate_a.projected,decoder_snapshot(att_branches[j],r*896),"async B_att vs separate workspace")
    exact_decoder(cache.key,decoder_snapshot(separate_cache.key,66*128),"async cache key")
    exact_decoder(cache.value,decoder_snapshot(separate_cache.value,66*128),"async cache value")
    cache.reset(ctx)
    assert_equal(cache.length,0)
    _ = _enqueue_policy_case(ctx,aw,cache,a,mw,m,TileTensor(xb,row_major(1,896)),
        Int(_case_mappings(variant,1,1)[2]) if variant >= 0 else 0,True,gqa,projection,variant)
    ctx.synchronize()
    check_decoder_active(m.output,name,"Y","full_slice",0,1,896)


def test_discriminating_negative_controls() raises:
    var name = String("h8_i12_nq2_nk1_d4_t7_s4001_base")
    var support = decoder_support()
    support.verify_case(name)
    var ctx = DeviceContext()
    var aw = AttentionWeights(ctx,2,1,4)
    var mw = MLPWeights(ctx,8,12)
    var a = AttentionWorkspace(ctx,7,7,2,1,4)
    var m = MLPWorkspace(ctx,7,8,12)
    _load_weights(aw,mw,a,name,7)
    var xb = ctx.enqueue_create_buffer[DType.bfloat16](7*8)
    load_decoder(xb,name,"input_X",0,7*8)
    var x = TileTensor(xb,row_major(7,8))
    support.configure(name,0,"full","negative")
    load_decoder(m.down,name,"full_B_mlp",0,7*8)
    enqueue_residual_apple_gpu(ctx,x,TileTensor(m.down,row_major(7,8)),TileTensor(m.output,row_major(7,8)))
    with assert_raises():
        check_decoder_slice(m.output,name,"Y",0,7,8)
    support.negative_passed("second residual uses X")
    with assert_raises():
        check_decoder_slice(m.down,name,"Y",0,7,8)
    support.negative_passed("second residual omitted")
    load_decoder(a.projected,name,"full_B_att",0,7*8)
    with assert_raises():
        check_decoder_slice(a.projected,name,"Z",0,7,8)
    support.negative_passed("first residual omitted")
    enqueue_mlp_stage_apple_gpu(ctx,mw,m,x,0,0)
    with assert_raises():
        check_decoder_slice(m.normalized,name,"N_mlp",0,7,8)
    support.negative_passed("second norm uses X")
    enqueue_rms_norm_apple_gpu(ctx,x,TileTensor(mw.norm,row_major(8)),TileTensor(a.normalized,row_major(7,8)))
    with assert_raises():
        check_decoder_slice(a.normalized,name,"N_att",0,7,8)
    support.negative_passed("wrong norm weights")
    load_decoder(a.raw_query,name,"full_Q_raw",0,7*8)
    enqueue_rope_apple_gpu(ctx,TileTensor(a.raw_query,row_major(1,2,4)),
        TileTensor(a.cosine,row_major(7,4)),TileTensor(a.sine,row_major(7,4)),
        TileTensor(a.query,row_major(1,2,4)),1)
    with assert_raises():
        check_decoder_slice(a.query,name,"Q",0,1,8)
    support.negative_passed("wrong absolute RoPE position")
    load_decoder(a.query,name,"full_Q",0,7*8)
    load_decoder(a.rotated_key,name,"full_K_rot",0,7*4)
    load_decoder(a.raw_value,name,"full_V_raw",0,7*4)
    # Supplying all keys with R=1 incorrectly marks query position zero as six.
    enqueue_grouped_query_attention_apple_gpu(ctx,TileTensor(a.query,row_major(1,2,4)),
        TileTensor(a.rotated_key,row_major(7,1,4)),TileTensor(a.raw_value,row_major(7,1,4)),
        TileTensor(a.fp32_scratch,row_major(1,2,7)),TileTensor(a.attention,row_major(1,2,4)))
    with assert_raises():
        check_decoder_slice(a.attention,name,"O",0,1,8)
    support.negative_passed("mask exposes future rows")
    var before = decoder_snapshot(a.rotated_key,7*4)
    a.rotated_key.enqueue_fill(0)
    with assert_raises():
        exact_decoder(a.rotated_key,before,"corrupted cache prefix")
    support.negative_passed("cache prefix changed")
