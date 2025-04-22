import json
import os
import csv
from algs.lib.traces import get_trace_reader, identify_trace
from algs.get_algorithm import get_algorithm
from algs.lib.cacheop import CacheOp

def run_miss_logger_from_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

    traces = config["traces"]
    cache_sizes = config["cache_sizes"]
    algorithms = config["algorithms"]
    request_count_type = config.get("request_count_type", "unique")
    associativity = config.get("associativity", 8)
    output_dir = config.get("output_miss", "miss_dir")

    os.makedirs(output_dir, exist_ok=True)

    def count_requests(trace_file):
        trace_type = identify_trace(trace_file)
        reader = get_trace_reader(trace_type)(trace_file)
        return reader.num_unique() if request_count_type == "unique" else reader.num_reuse()

    csv_header = [
        "trace", "algorithm", "cache_size", "accesses", "hits", "misses",
        "hit_rate(%)", "compulsory", "capacity", "conflict"
    ]

    summary_file = os.path.join(output_dir, "summary.csv")
    with open(summary_file, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)

    for trace_path in traces:
        if not os.path.exists(trace_path):
            print(f"Trace not found: {trace_path}")
            continue

        trace_type = identify_trace(trace_path)
        trace_reader_cls = get_trace_reader(trace_type)

        for cache_size in cache_sizes:
            if isinstance(cache_size, float):
                count = count_requests(trace_path)
                actual_cache_size = int(cache_size * count)
            else:
                actual_cache_size = cache_size

            if actual_cache_size < 1:
                print(f" Cache size {actual_cache_size} too small. Skipping.")
                continue

            for algorithm in algorithms:
                trace_reader = trace_reader_cls(trace_path)
                AlgClass = get_algorithm(algorithm)
                if AlgClass is None:
                    print(f"Skipping unknown algorithm: {algorithm}")
                    continue

                try:
                    cache = AlgClass(actual_cache_size, window_size=10, enable_visual=False)
                except Exception as e:
                    print(f"Error creating algorithm {algorithm}: {e}")
                    continue

                # Init counters
                compulsory = capacity = conflict = hits = misses = 0
                seen_blocks = set()
                total_accesses = 0

                for lba, _, _ in trace_reader.read():
                    total_accesses += 1
                    first_time = lba not in seen_blocks
                    if first_time:
                        seen_blocks.add(lba)

                    op, evicted = cache.request(lba, ts=None)

                    if op == CacheOp.HIT:
                        hits += 1
                    else:
                        misses += 1
                        if first_time:
                            compulsory += 1
                        elif evicted is not None:
                            # Heuristics for miss type
                            if hasattr(cache, 'lfu') and evicted in cache.lfu:
                                conflict += 1
                            else:
                                capacity += 1

                # Compute final metrics
                hit_rate = round((hits / total_accesses) * 100, 2)
                trace_name = os.path.basename(trace_path).split('.')[0]

                # Write to summary CSV
                with open(summary_file, "a", newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        trace_name, algorithm, actual_cache_size, total_accesses,
                        hits, misses, hit_rate, compulsory, capacity, conflict
                    ])

                print(f"{trace_name} | {algorithm.upper()} | Cache={actual_cache_size} | HitRate={hit_rate}%")

    print(f"\nSummary CSV written to: {summary_file}")


if __name__ == "__main__":
    print("Running miss_logger using example.config ...")
    run_miss_logger_from_config("example.config")
