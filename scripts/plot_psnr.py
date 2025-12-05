import argparse
import subprocess
import requests
import time
import socket
import os
import shutil
import tempfile
from urllib.parse import quote
import matplotlib.pyplot as plt

def find_free_port():
    s = socket.socket()
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return port

def wait_for_tb(port, timeout=50):
    url = f"http://localhost:{port}/data/logdir"
    start = time.time()
    while True:
        try:
            r = requests.get(url)
            if r.status_code == 200:
                return
        except Exception:
            pass
        if time.time() - start > timeout:
            raise RuntimeError("TensorBoard server did not start in time.")
        time.sleep(0.5)

def fetch_scalar(tb_url, run, tag):
    tag_encoded = quote(tag, safe='')
    run_encoded = quote(run, safe='')

    url = f"{tb_url}/data/plugin/scalars/scalars?tag={tag_encoded}&run={run_encoded}"
    r = requests.get(url)
    r.raise_for_status()
    return r.json()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir1", help="First TensorBoard log directory")
    parser.add_argument("dir2", help="Second TensorBoard log directory")
    parser.add_argument("output", help="Path to save the figure, e.g., /tmp/psnr_plot.png")
    parser.add_argument("dataset_name", help="Name of the dataset")
    args = parser.parse_args()

    tag = "test/loss_viewpoint - psnr"

    # Create temp parent directory
    temp_parent = tempfile.mkdtemp(prefix="tb_temp_parent_")
    run1_name = os.path.basename(os.path.normpath(args.dir1))
    run2_name = os.path.basename(os.path.normpath(args.dir2))
    temp_dir1 = os.path.join(temp_parent, run1_name)
    temp_dir2 = os.path.join(temp_parent, run2_name)

    print(f"Copying logdirs to temporary parent directory: {temp_parent}")
    shutil.copytree(args.dir1, temp_dir1)
    shutil.copytree(args.dir2, temp_dir2)

    port = find_free_port()
    print(f"Starting TensorBoard on port {port}...")

    tb = subprocess.Popen([
        "tensorboard",
        "--logdir", temp_parent,
        "--port", str(port),
    ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    try:
        wait_for_tb(port)
        print("TensorBoard ready. Querying JSON...")

        tb_url = f"http://localhost:{port}"
        runs = [run1_name, run2_name]

        plt.figure(figsize=(8,5))

        for run in runs:
            data = fetch_scalar(tb_url, run, tag)
            if not data:
                print(f"Warning: No data found for tag '{tag}' in run '{run}'")
                continue

            abs_times = [d[0] for d in data]
            values = [d[2] for d in data]
            t0 = abs_times[0]
            times = [t - t0 for t in abs_times]

            plt.plot(times, values, label=run, linewidth=2)

        plt.xlabel("Relative Time (s)")
        plt.ylabel("PSNR")
        plt.title(args.dataset_name)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(args.output)
        print(f"Figure saved to {args.output}")

    finally:
        tb.terminate()
        try:
            tb.wait(timeout=5)
        except subprocess.TimeoutExpired:
            tb.kill()
        # Delete temporary directory
        shutil.rmtree(temp_parent)
        print(f"Temporary directory {temp_parent} deleted.")

if __name__ == "__main__":
    main()

