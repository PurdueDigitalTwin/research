import os
import subprocess  # nosec B404

import jax


def setup_jax_distributed() -> None:
    r"""Auto-detects the distributed environment and initialize JAX runtime."""

    coordinator_address = None
    num_processes = None
    process_id = None
    local_device_ids = None

    if isinstance(os.getenv("MASTER_ADDR"), str):
        # case 1: launched with MPI (e.g., using `mpirun` or `srun`)
        master_address = os.getenv("MASTER_ADDR")
        port = int(os.getenv("MASTER_PORT", "12345"))
        num_processes = int(os.getenv("WORLD_SIZE", "1"))
        process_id = int(os.getenv("RANK", "0"))
        coordinator_address = f"{master_address}:{port}"

        if isinstance(os.getenv("LOCAL_RANK"), str):
            local_rank = str(os.getenv("LOCAL_RANK"))
            local_device_ids = [int(i) for i in local_rank.split(",")]

    elif isinstance(os.getenv("SLURM_PROCID"), str):
        # case 2: launched with SLURM HPC scheduler
        try:
            # try resolve the master address
            nodelist = os.getenv("SLURM_JOB_NODELIST", "")
            master_node = (
                subprocess.check_output(  # nosec B603
                    args=["scontrol", "show", "hostnames", nodelist]
                )
                .decode("utf-8")
                .splitlines()[0]
            )
        except (subprocess.CalledProcessError, KeyError, IndexError):
            master_node = os.getenv("SLURM_LAUNCH_NODE_IPADDR", "localhost")

        port = int(os.getenv("MASTER_PORT", "12345"))
        num_processes = int(os.getenv("SLURM_NTASKS", "1"))
        process_id = int(os.getenv("SLURM_PROCID", "0"))
        coordinator_address = f"{master_node}:{port}"

        if isinstance(os.getenv("SLURM_LOCALID"), str):
            local_id = str(os.getenv("SLURM_LOCALID"))
            local_device_ids = [int(local_id)]

    if coordinator_address:
        jax.distributed.initialize(
            coordinator_address=coordinator_address,
            num_processes=num_processes,
            process_id=process_id,
            local_device_ids=local_device_ids,
        )
    else:
        jax.distributed.initialize()
