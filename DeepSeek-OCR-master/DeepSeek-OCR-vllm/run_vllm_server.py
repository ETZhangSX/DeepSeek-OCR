import os
import torch
import argparse

# 设置环境变量（参考 run_dpsk_ocr_pdf.py）
if torch.version.cuda == '11.8':
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"
os.environ['VLLM_USE_V1'] = '0'

# 导入配置
from config import MODEL_PATH, MAX_CONCURRENCY

# 导入模型相关
from deepseek_ocr import DeepseekOCRForCausalLM
from vllm.model_executor.models.registry import ModelRegistry
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor

# 注册模型（必须在导入 vllm.entrypoints 之前完成）
ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="启动 DeepSeek-OCR vLLM API Server")
    
    # 模型相关参数
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_PATH,
        help="模型路径（默认从 config.py 读取）"
    )
    
    # 服务器相关参数
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="服务器监听地址（默认: 0.0.0.0）"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="服务器端口（默认: 8000）"
    )
    
    parser.add_argument(
        "--served-model-name",
        type=list,
        default=None,
    )
    
    # GPU 相关参数
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU 内存使用率（默认: 0.9）"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="张量并行大小（默认: 1）"
    )
    parser.add_argument(
        "--cuda-visible-devices",
        type=str,
        default="0",
        help="可见的 CUDA 设备（默认: 0）"
    )
    
    # 模型配置参数
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="最大模型长度（默认: 8192）"
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=256,
        help="块大小（默认: 256）"
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=MAX_CONCURRENCY,
        help="最大并发序列数（默认从 config.py 读取）"
    )
    parser.add_argument(
        "--swap-space",
        type=int,
        default=0,
        help="交换空间大小（默认: 0）"
    )
    
    # 其他参数
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="信任远程代码（默认启用）"
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="强制使用 eager 模式"
    )
    parser.add_argument(
        "--disable-mm-preprocessor-cache",
        action="store_true",
        help="禁用多模态预处理器缓存（默认启用）"
    )
    
    return parser.parse_args()

logits_processors = [NoRepeatNGramLogitsProcessor(ngram_size=20, window_size=50, whitelist_token_ids= {128821, 128822})] 


def main():
    """主函数：启动 vLLM API Server"""
    args = parse_args()

    engine_args = AsyncEngineArgs(
        model=args.model,
        served_model_name=args.served_model_name,
        hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
        block_size=args.block_size,
        trust_remote_code=args.trust_remote_code,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        swap_space=args.swap_space,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        logits_processors=logits_processors,
        disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache,
    )

    engine = AsyncLLMEngine.from_engine_args(engine_args)
    
    print("=" * 60)
    print("启动 DeepSeek-OCR vLLM API Server")
    print("=" * 60)
    print(f"模型路径: {args.model}")
    print(f"模型路由名: {args.served_model_name}")
    print(f"服务器地址: http://{args.host}:{args.port}")
    print(f"GPU 内存使用率: {args.gpu_memory_utilization}")
    print(f"最大并发序列数: {args.max_num_seqs}")
    print("=" * 60)
    
    try:
        # 导入并启动 API server（必须在模型注册之后）
        from vllm.entrypoints.api_server import run_server
        run_server(args, engine)
    except KeyboardInterrupt:
        print("\n服务器已停止")
    except Exception as e:
        print(f"启动服务器时出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

