"""
Based on:
https://www.jinghong-chen.net/estimate-vram-usage-in-llm-inference/
https://linden-li.github.io/posts/inference-slides?ref=jinghong-chen.net

For training:
https://blog.eleuther.ai/transformer-math/


IMPORTANT:
Communication times between gpus increase latency quite a bit. In mistral7b for bs=1 in=100 out=4 it goes from 15.90ms
with 1gpu to 220.90ms with 2 gpus
"""
import matplotlib.pyplot as plt

def generate_plot(batch_sizes, values, value_name, gpu_count, tokens_in, tokens_out, model_name, gpu_name):
    plt.figure(figsize=(10, 6))
    plt.plot(batch_sizes, values)
    plt.xlabel('Batch Size')
    plt.ylabel(value_name)
    plt.title(f'{model_name} - {value_name} vs Batch Size In: {tokens_in} Out: {tokens_out} ({gpu_count}x{gpu_name})')
    plt.grid(True)
    plt.savefig(f'plots/{value_name}-gpus={gpu_count}.png')
    plt.close()

def get_prefill_compute_flops(model_params, batch_size, sequence_length, byte_per_param):
    return byte_per_param * model_params * batch_size * sequence_length

def get_decode_compute_flops(model_params, batch_size, byte_per_param):
    return byte_per_param * model_params * batch_size * 1

def get_prefill_memory(model_params, byte_per_param):
    return byte_per_param * model_params

def get_decode_memory(model_params, byte_per_param):
    return get_prefill_memory(model_params, byte_per_param)

def get_ttft(prefill_compute_flops, prefill_memory, num_gpus, gpu_flop_rate, gpu_hbm_rate):
    prefill_compute_time = prefill_compute_flops / (num_gpus * gpu_flop_rate)
    prefill_memory_time = prefill_memory / (num_gpus * gpu_hbm_rate)

    return {
        "prefill_compute_time": prefill_compute_time,
        "prefill_memory_time": prefill_memory_time,
        "ttft": max(prefill_compute_time, prefill_memory_time),
    }

def get_tpot(decode_compute_flops, decode_memory, num_gpus, gpu_flop_rate, gpu_hbm_rate):
    decode_compute_time = decode_compute_flops / (num_gpus * gpu_flop_rate)
    decode_memory_time = decode_memory / (num_gpus * gpu_hbm_rate)

    return {
        "decode_compute_time": decode_compute_time,
        "decode_memory_time": decode_memory_time,
        "tpot": max(decode_compute_time, decode_memory_time),
    }

def get_vram_size(byte_per_param, batch_size, sequence_length, output_length, model_info):
    # kv_cache =
    #   2 (key&value vectors) * #layers (L) * KV_heads (num_kv_heads) * head dimension (kv_heads_dim) * bs * (tokens_in + tokens_out)
    if "kv_heads_dim" in model_info:
        kv_cache_params = 2 * model_info["num_layers"] * model_info["num_kv_heads"] * model_info["kv_heads_dim"] * batch_size * (sequence_length + output_length)
    else:
        kv_cache_params = 2 * model_info["hidden_dim"] * model_info["num_layers"] * batch_size * (sequence_length + output_length)

    kv_cache_vram = byte_per_param * kv_cache_params
    # take into account factor of 20% for model params
    # https://blog.eleuther.ai/transformer-math/
    model_vram = byte_per_param * model_info["param_count"] * 1.2
    return {
        "total_vram": kv_cache_vram + model_vram,
        "model_vram": model_vram,
        "kv_cache_vram": kv_cache_vram,
    }


if __name__ == "__main__":

    GPU_UTILIZATION_RATE = 0.7
    gpus = {
        "NVIDIA_A100_80GB_PCI": {
                "flops_rate": 312 * pow(10, 12) * GPU_UTILIZATION_RATE,
                "hbm_rate": 1.5 * pow(10, 12) * GPU_UTILIZATION_RATE
        },
    }

    models = {
        "Mistral7b": {
            "param_count": 7 * pow(10, 9),
            "kv_heads_dim": 128,
            "num_kv_heads": 8,
            "num_layers": 32,
            "hidden_dim": 4096
        },
        "LLaMa7b": {
            "param_count": 7 * pow(10, 9),
            "num_layers": 32,
            "hidden_dim": 4096
        },
    }

    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    num_gpus = [1]

    model_name = "Mistral7b"
    gpu_name = "NVIDIA_A100_80GB_PCI"
    bit_precision = 16
    bytes_per_param = bit_precision / 8
    model, gpu = models[model_name], gpus[gpu_name]

    tokens_in, tokens_out = 500, 4

    print("###\t Milliseconds per token ###")
    print(f"GPU: {gpu_name}\tModel: {model_name}\tTokens in: {tokens_in}\tTokens out: {tokens_out}")


    for gpu_count in num_gpus:
        plot_values = {
            "total_vram": [],
            "model_vram": [],
            "kv_cache_vram": [],
            "prefill_memory_time": [],
            "prefill_compute_time": [],
            "ttft": [],
            "decode_memory_time": [],
            "decode_compute_time": [],
            "tpot": [],
            "total_decode_time": [],
            "total_time": [],
        }
        print("#"*100)
        for batch_size in batch_sizes:
            vram_dict = get_vram_size(
                model_info=model,
                byte_per_param=bytes_per_param,
                batch_size=batch_size,
                sequence_length=tokens_in,
                output_length=tokens_out,
            )

            for key, size in vram_dict.items():
                vram_dict[key] = size * pow(10, -6)
                plot_values[key].append(vram_dict[key])

            print(
                f"#GPUs: {gpu_count}\t Batch size: {batch_size}\t VRAM Total Size: {vram_dict['total_vram']:.3f}MB\t"
                f"VRAM Model Size: {vram_dict['model_vram']:.3f}MB\t"
                f"VRAM KV Cache Size: {vram_dict['kv_cache_vram']:.3f}MB\t"
                f"VRAM KV Cache Per Token Size: {vram_dict['kv_cache_vram'] / (batch_size * (tokens_in+tokens_out)):.3f}MB\t"
            )

            prefill_memory = get_prefill_memory(
                model_params=model["param_count"],
                byte_per_param=bytes_per_param
            )
            prefill_compute_flops = get_prefill_compute_flops(
                model_params=model["param_count"],
                batch_size=batch_size,
                sequence_length=tokens_in,
                byte_per_param=bytes_per_param
            )
            prefill_times = get_ttft(
                prefill_memory=prefill_memory,
                prefill_compute_flops=prefill_compute_flops,
                num_gpus=gpu_count,
                gpu_flop_rate=gpu["flops_rate"],
                gpu_hbm_rate=gpu["hbm_rate"]
            )

            decode_memory = get_decode_memory(
                model_params=model["param_count"],
                byte_per_param=bytes_per_param
            )
            decode_compute_flops = get_decode_compute_flops(
                model_params=model["param_count"],
                batch_size=batch_size,
                byte_per_param=bytes_per_param
            )
            decode_times = get_tpot(
                decode_memory=decode_memory,
                decode_compute_flops=decode_compute_flops,
                num_gpus=gpu_count,
                gpu_flop_rate=gpu["flops_rate"],
                gpu_hbm_rate=gpu["hbm_rate"]
            )

            decode_memory_time_ms = decode_times["decode_memory_time"] * 1000
            decode_compute_time_ms = decode_times["decode_compute_time"] * 1000
            tpot = decode_times["tpot"] * 1000
            total_decode_time = (tpot*tokens_out)/1000

            plot_values["decode_memory_time"].append(decode_memory_time_ms)
            plot_values["decode_compute_time"].append(decode_compute_time_ms)
            plot_values["tpot"].append(tpot)
            plot_values["total_decode_time"].append(total_decode_time)

            decode_bound_type = 'Memory' if decode_memory_time_ms > decode_compute_time_ms else 'Compute'

            prefill_memory_time_ms = prefill_times["prefill_memory_time"] * 1000
            prefill_compute_time_ms = prefill_times["prefill_compute_time"] * 1000
            ttft = prefill_times["ttft"] * 1000 + tpot # have to add the time that takes to generate that 1st token
            plot_values["prefill_memory_time"].append(prefill_memory_time_ms)
            plot_values["prefill_compute_time"].append(prefill_compute_time_ms)
            plot_values["ttft"].append(ttft)

            prefill_bound_type = 'Memory' if prefill_memory_time_ms > prefill_compute_time_ms else 'Compute'

            print(f"\tPrefill Memory time: {prefill_memory_time_ms:.2f}ms"
                  f"\tPrefill Compute time: {prefill_compute_time_ms:.2f}ms" 
                  f"\t{prefill_bound_type} Bound" 
                  f"\tTFTT: {ttft:.2f}ms"
            )

            print(f"\tDecode Memory time: {decode_memory_time_ms:.2f}ms"
                  f"\tDecode Compute time: {decode_compute_time_ms:.2f}ms"
                  f"\t{decode_bound_type} Bound"
                  f"\tTPOT: {tpot:.2f}ms"
                  f"\tTotal decode time: {total_decode_time:.2f}s"
                  )

            total_time = ttft/1000 + total_decode_time - tpot/1000 # the 1st token is already in ttft
            plot_values["total_time"].append(total_time)

            print(f"\tTotal time of {tokens_in+tokens_out} tokens: {total_time:.2f}s")
            print("-"*50)

        for value_name, values in plot_values.items():
            generate_plot(
                batch_sizes=batch_sizes,
                values=values,
                value_name=value_name,
                gpu_count=gpu_count,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                model_name=model_name,
                gpu_name=gpu_name
            )
