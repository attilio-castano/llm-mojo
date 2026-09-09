"""Identical-operand native operation capture for an exposed development case."""
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from llm_mojo.model import ModelLayer, load_bf16, save_bf16
from llm_mojo.attention_sublayer import AttentionWorkspace, _enqueue_attention_qkv, _enqueue_attention_wo
from llm_mojo.attention_decode import enqueue_grouped_query_attention_consistent_apple_gpu
from llm_mojo.mlp import MLPWorkspace, enqueue_mlp_stage_apple_gpu
from llm_mojo.rms_norm import enqueue_rms_norm_apple_gpu
from llm_mojo.residual import enqueue_residual_apple_gpu


def capture_operations(ctx: DeviceContext, prepared: String, reference: String, output: String) raises:
    print("operation device",ctx.name(),"backend",ctx.api())
    var a = AttentionWorkspace(ctx,1,1,14,2,64,False,False)
    var m = MLPWorkspace(ctx,1)
    var xb = ctx.enqueue_create_buffer[DType.bfloat16](896)
    for i in range(24):
        var layer = ModelLayer(ctx,prepared,i,1)
        var src = reference+"/layer_"+String(i)+"_"
        var dst = output+"/layer_"+String(i)+"_"
        load_bf16(xb,src+"X.bin")
        enqueue_rms_norm_apple_gpu(ctx,TileTensor(xb,row_major(1,896)),
            TileTensor(layer.attention.norm,row_major(896)),TileTensor(a.normalized,row_major(1,896)))
        save_bf16(a.normalized,dst+"N_att.bin",896)
        load_bf16(a.normalized,src+"N_att.bin")
        _enqueue_attention_qkv(ctx,layer.attention,a,1,0)
        save_bf16(a.raw_query,dst+"Q_raw.bin",896)
        save_bf16(a.raw_key,dst+"K_raw.bin",128)
        save_bf16(a.raw_value,dst+"V_raw.bin",128)
        # Absolute position zero has identity RoPE. Use actual upstream Q/K/V.
        load_bf16(a.raw_query,src+"Q_raw.bin")
        load_bf16(a.raw_key,src+"K_raw.bin")
        load_bf16(a.raw_value,src+"V_raw.bin")
        enqueue_grouped_query_attention_consistent_apple_gpu(ctx,
            TileTensor(a.raw_query,row_major(1,14,64)),TileTensor(a.raw_key,row_major(1,2,64)),
            TileTensor(a.raw_value,row_major(1,2,64)),TileTensor(a.attention,row_major(1,14,64)),
            TileTensor(a.split,row_major(14,1,66)))
        save_bf16(a.attention,dst+"O.bin",896)
        load_bf16(a.attention,src+"O.bin")
        _enqueue_attention_wo(ctx,layer.attention,a,1,False,0)
        save_bf16(a.projected,dst+"B_att.bin",896)
        load_bf16(a.projected,src+"B_att.bin")
        enqueue_residual_apple_gpu(ctx,TileTensor(xb,row_major(1,896)),
            TileTensor(a.projected,row_major(1,896)),TileTensor(a.output,row_major(1,896)))
        save_bf16(a.output,dst+"Z.bin",896)
        load_bf16(xb,src+"Z.bin")
        var x = TileTensor(xb,row_major(1,896))
        enqueue_mlp_stage_apple_gpu(ctx,layer.mlp,m,x,0,0)
        save_bf16(m.normalized,dst+"N_mlp.bin",896)
        load_bf16(m.normalized,src+"N_mlp.bin")
        enqueue_mlp_stage_apple_gpu(ctx,layer.mlp,m,x,1,0)
        save_bf16(m.gate,dst+"G.bin",4864)
        enqueue_mlp_stage_apple_gpu(ctx,layer.mlp,m,x,2,0)
        save_bf16(m.up,dst+"U.bin",4864)
        load_bf16(m.gate,src+"G.bin")
        enqueue_mlp_stage_apple_gpu(ctx,layer.mlp,m,x,3,0)
        save_bf16(m.activated,dst+"A.bin",4864)
        load_bf16(m.activated,src+"A.bin")
        load_bf16(m.up,src+"U.bin")
        enqueue_mlp_stage_apple_gpu(ctx,layer.mlp,m,x,4,0)
        save_bf16(m.gated,dst+"S.bin",4864)
        load_bf16(m.gated,src+"S.bin")
        enqueue_mlp_stage_apple_gpu(ctx,layer.mlp,m,x,5,0)
        save_bf16(m.down,dst+"B_mlp.bin",896)
        load_bf16(m.down,src+"B_mlp.bin")
        enqueue_mlp_stage_apple_gpu(ctx,layer.mlp,m,x,6,0)
        save_bf16(m.output,dst+"Y.bin",896)
        print("completed identical-operand layer",i)
