import argparse
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse timing results from a log file.")
    parser.add_argument("log_file", type=str, help="Path to the log file containing timing results.")
    args = parser.parse_args()

    ns = []
    forward_timings = []
    jvp_timings = []
    primal_timings = []
    dual_timings = []

    forward_mems = []
    jvp_mems = []
    primal_mems = []
    dual_mems = []

    with open(args.log_file, 'r') as f:
        for line in f:
            if "num_images" in line: 
                num_images = line.split()[-1]
                print(f"Number of images: {num_images}")
                ns.append(int(num_images))

            if "Forward time ms:" in line:
                forward_time = float(line.split()[-2])
                forward_timings.append(forward_time)

            if "JVP time ms:" in line:
                jvp_time = float(line.split()[-2])
                jvp_timings.append(jvp_time)

            if "Primal time ms:" in line:
                primal_time = float(line.split()[-2])
                primal_timings.append(primal_time)

            if "Dual time ms:" in line:
                dual_time = float(line.split()[-2])
                dual_timings.append(dual_time)

            if "Forward Peak memory usage (MB):" in line:
                forward_mem = float(line.split()[-2])
                forward_mems.append(forward_mem)

            if "JVP Peak memory usage (MB):" in line:
                jvp_mem = float(line.split()[-2])
                jvp_mems.append(jvp_mem)

            if "Primal Peak memory usage (MB):" in line:
                primal_mem = float(line.split()[-2])
                primal_mems.append(primal_mem)

            if "Dual Peak memory usage (MB):" in line:
                dual_mem = float(line.split()[-2])
                dual_mems.append(dual_mem)

    plt.plot(ns, forward_timings, label='Forward', marker='o')
    plt.plot(ns, jvp_timings, label='Forward + JVP', marker='o')

    plt.legend()
    plt.xlabel('Number of Images')
    plt.ylabel('Time (ms)')
    plt.grid(True)

    plt.savefig('figures/jvp_timing_results.png', bbox_inches='tight')

    plt.close()

    plt.plot(ns, forward_mems, label='Forward', marker='o')
    plt.plot(ns, jvp_mems, label='Forward + JVP', marker='o')

    plt.ylim(0, max(jvp_mems) + 500)
    
    plt.legend()

    plt.xlabel('Number of Images')
    plt.ylabel('Peak GPU Memory Usage (MB)')
    plt.grid(True)
    plt.savefig('figures/jvp_memory_results.png', bbox_inches='tight')

    plt.close()

    plt.plot(ns, primal_timings, label='Forward + Backward', marker='o')
    plt.plot(ns, dual_timings, label='Forward + HVP', marker='o')

    plt.legend()
    plt.xlabel('Number of Images')
    plt.ylabel('Time (ms)')
    plt.grid(True)

    plt.savefig('figures/hvp_timing_results.png', bbox_inches='tight')
    plt.close()

    plt.plot(ns, primal_mems, label='Forward + Backward', marker='o')
    plt.plot(ns, dual_mems, label='Forward + HVP', marker='o')
    plt.ylim(0, max(dual_mems) + 500)
    plt.legend()

    plt.xlabel('Number of Images')
    plt.ylabel('Peak GPU Memory Usage (MB)')
    plt.grid(True)

    plt.savefig('figures/hvp_memory_results.png', bbox_inches='tight')
    plt.close()
