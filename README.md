# Parallel Word Counting in a Text Corpus with MPI

## 1. Title

Parallel Word Counting in a Text Corpus with MPI — Lab 2, Estructura del Computador II, Universidad del Norte.

## 2. Problem Description

The generator (`generator.py`, seed 42) creates a `dataset/` folder with 3,000 text files (`file_0001.txt` … `file_3000.txt`, about 45 million words in total) and a `consulta.txt` file with 200 query words. The goal is to count how many times each query word appears in the whole corpus and report the 10 most frequent ones.

The files have very different sizes: about 60% are small (2,000–5,000 words), 30% medium (10,000–20,000) and 10% large (50,000–100,000). Because of this, giving every process the same number of files does not mean giving it the same amount of work, which is the main problem this lab studies.

## 3. Environment and Execution Instructions

- Machine: MacBook Air, Apple M4, 10 cores visible inside the container
- Container: `augustosalazar/slim-mpi:2` (Python 3 + Open MPI + mpi4py)
- Number of processes: p = 8 (value reported by `detectar_procesos_mpi.py` on this machine)

All commands are run from the root folder of the repository.

Generate the dataset (macOS / Linux):

```bash
docker run --rm -v "$(pwd)":/app augustosalazar/slim-mpi:2 python /app/generator.py
```

Windows (Command Prompt):

```bash
docker run --rm -v "%cd%:/app" augustosalazar/slim-mpi:2 python /app/generator.py
```

### a. Experimental Plan

1. Generate the dataset once with `generator.py` (fixed seed, so it is always the same).
2. Run the sequential baseline 3 times and take the average time as T_seq.
3. Run MPI version 1 three times with p = 8, recording the total time and the local time of each process.
4. Run MPI version 2 three times with the same p = 8.
5. Check that both MPI versions produce exactly the same counts as the baseline.
6. Compute T_p (average total time), speedup S_p = T_seq / T_p, efficiency E_p = S_p / p and load imbalance LI = max(t_i) / mean(t_i).

All runs were done on the same machine, with the same dataset and the same container, with the laptop plugged in and other applications closed. The output of every run is saved in the `logs/` folder.

### b. Sequential baseline (what it does and how)

`baseline_secuencial.py` (provided, not modified) reads `consulta.txt` into a set, then opens every `file_*.txt` one by one, splits each line into words, converts them to lowercase and adds 1 to a `Counter` every time a word belongs to the query set. It prints the execution time and the top 10, and saves the full count in `dataset/baseline_results.csv`.

```bash
docker run --rm -v "$(pwd)":/app augustosalazar/slim-mpi:2 python /app/baseline_secuencial.py
```

### c. MPI version 1 (what it does and how)

`mpi1.py` follows the steps of the assignment:

1. Rank 0 reads `consulta.txt` (reusing `cargar_consulta` from the baseline).
2. Rank 0 sends the query words to all processes with `comm.bcast`.
3. Rank 0 lists the `file_*.txt` files in sorted order.
4. The files are split statically in contiguous blocks with the same number of files per process (3,000 / 8 = 375) and sent with `comm.scatter`.
5. Each process counts the query words in its own files with the same logic as the baseline, measuring its local time.
6. The partial `Counter`s and local times are collected in rank 0 with `comm.gather`.
7. Rank 0 adds the partial counters, prints the top 10, the per-process report (files, MB and local time) and the LI.

The total time is measured with `MPI.Wtime()` from a `Barrier` at the start until rank 0 has the global result.

```bash
docker run --rm -v "$(pwd)":/app augustosalazar/slim-mpi:2 mpiexec --allow-run-as-root --use-hwthread-cpus --oversubscribe -n 8 python /app/mpi1.py
```

### d. Test procedure

1. Correctness: both MPI versions save their global count in `dataset/mpi1_results.csv` and `dataset/mpi2_results.csv` with the same format as the baseline, and they are compared with:

```bash
diff dataset/baseline_results.csv dataset/mpi1_results.csv
diff dataset/baseline_results.csv dataset/mpi2_results.csv
```

Both comparisons produced no differences, and the top 10 of every run is the same as the baseline.

2. Timing: each program was run 3 times, saving the output with `tee`, for example:

```bash
docker run --rm -v "$(pwd)":/app augustosalazar/slim-mpi:2 mpiexec --allow-run-as-root --use-hwthread-cpus --oversubscribe -n 8 python /app/mpi1.py | tee logs/mpi1_run1.txt
```

The `run_all.sh` script (provided, not modified) can also run the whole experiment automatically:

```bash
docker run --rm -e NP_LIST=8 -v "$(pwd)":/app augustosalazar/slim-mpi:2 sh /app/run_all.sh
```

## 4. Experimental Plan Execution

### a. Sequential baseline timing

| Run | Time (s) |
|---|---:|
| 1 | 5.343 |
| 2 | 6.268 |
| 3 | 5.410 |
| **Average (T_seq)** | **5.673** |

Global output (same in every run): 3,000 files, 44,951,458 tokens, 3,631,778 occurrences. Top 10: a: 785774, para: 392156, sus: 228913, otros: 105530, ante: 99832, unos: 88794, otra: 83901, vosotros: 61617, mios: 58420, tuya: 56635.

### b. MPI version 1 timing results

| Run | Total time (s) | LI |
|---|---:|---:|
| 1 | 1.789 | 1.058 |
| 2 | 1.907 | 1.072 |
| 3 | 1.795 | 1.074 |
| **Average** | **1.830** | **1.068** |

Speedup S_8 = 5.673 / 1.830 = **3.10** ; Efficiency E_8 = 3.10 / 8 = **0.39**

### c. Load imbalance evidence (per-process local times and LI)

Run 1 of version 1:

| Rank | Files | MB | Local time (s) |
|---|---:|---:|---:|
| 0 | 375 | 36.73 | 1.357 |
| 1 | 375 | 29.37 | 1.218 |
| 2 | 375 | 37.82 | 1.438 |
| 3 | 375 | 36.57 | 1.460 |
| 4 | 375 | 32.32 | 1.309 |
| 5 | 375 | 33.86 | 1.396 |
| 6 | 375 | 33.10 | 1.402 |
| 7 | 375 | 35.28 | 1.460 |

All processes received 375 files, but the blocks have different sizes: from 29.37 MB (rank 1) to 37.82 MB (rank 2), almost 30% more data. The local times follow the same pattern: rank 1 finished first (1.218 s) and ranks 3 and 7 were the slowest (1.460 s). In this run LI = 1.058 and the relative spread (max − min) / mean = 0.175. The average LI over the 3 runs was 1.068.

The per-process local times of all runs are saved in `logs/mpi1_run1.txt` , `logs/mpi2_run3.txt`, and the complementary `run_all.sh` logs in `logs/run_all/`.

### d. MPI version 2: how it corrects the imbalance, and its results

Version 2 (`mpi2.py`) keeps exactly the same flow as version 1 and only changes how the files are split. Instead of giving the same number of files to each process, rank 0 balances the number of bytes: it gets the size of every file, sorts the files from biggest to smallest and gives each one to the process that has the fewest bytes so far (greedy Longest Processing Time first). The big files are placed first and the small ones fill the gaps, so every process ends with almost the same amount of data.

```bash
docker run --rm -v "$(pwd)":/app augustosalazar/slim-mpi:2 mpiexec --allow-run-as-root --use-hwthread-cpus --oversubscribe -n 8 python /app/mpi2.py
```

| Run | Total time (s) | LI |
|---|---:|---:|
| 1 | 1.882 | 1.023 |
| 2 | 1.823 | 1.011 |
| 3 | 1.459 | 1.106 |
| **Average** | **1.721** | **1.047** |

Speedup S_8 = 5.673 / 1.721 = **3.30** ; Efficiency E_8 = 3.30 / 8 = **0.41**

Per-process data of run 1:

| Rank | Files | MB | Local time (s) |
|---|---:|---:|---:|
| 0 | 376 | 34.38 | 1.478 |
| 1 | 375 | 34.38 | 1.424 |
| 2 | 374 | 34.38 | 1.471 |
| 3 | 375 | 34.38 | 1.442 |
| 4 | 375 | 34.38 | 1.445 |
| 5 | 375 | 34.38 | 1.461 |
| 6 | 375 | 34.38 | 1.442 |
| 7 | 375 | 34.38 | 1.490 |

Now every process has the same 34.38 MB (the number of files changes a little, from 374 to 376). In run 1 the local times are between 1.424 s and 1.490 s, with LI = 1.023 and relative spread = 0.045 (it was 0.175 in version 1).

Comparison (p = 8):

| Version | T_p (s) | Speedup | Efficiency | LI |
|---|---:|---:|---:|---:|
| Sequential | 5.673 | 1.00 | — | — |
| MPI v1 (static blocks) | 1.830 | 3.10 | 0.39 | 1.068 |
| MPI v2 (size-aware) | 1.721 | 3.30 | 0.41 | 1.047 |

## 5. Analysis

### a. Parallel performance

MPI version 1 did improve on the sequential baseline: the time went from 5.673 s to 1.830 s, a speedup of 3.10. However, this is far from the ideal S_8 = 8, with an efficiency of only 0.39.

The gap is not mainly caused by load imbalance. Two factors explain it:

1. **Non-parallel work.** In run 1 the slowest process finished its local count in 1.460 s, but the total time was 1.789 s. About 0.33 s (18% of the total) is spent outside the parallel part: listing the files, `bcast`, `scatter`, `gather` and merging the counters in rank 0.
2. **Slower processes when running together.** The sum of the 8 local times in run 1 is 11.04 s, almost twice the 5.67 s that a single process needs. Running 8 processes at once means they compete for the disk (through the Docker shared folder), memory and cache, and some of them run on the slower efficiency cores of the M4.

### b. Load balance

Version 1 showed a clear imbalance in data, but a small one in time. All processes received 375 files, yet the blocks ranged from 29.37 MB to 37.82 MB, and in run 1 the local times went from 1.218 s to 1.460 s (average LI = 1.068).

Version 2 corrected the data imbalance: every process received 34.38 MB, and in run 1 the LI dropped to 1.023 and the relative spread from 0.175 to 0.045.

In our main experiment this did not produce a clear performance improvement. The average time of version 2 is 6% lower (1.721 s vs 1.830 s), but this comes mostly from run 3 (1.459 s). Runs 1 and 2 of version 2 (1.882 s and 1.823 s) are in the same range as the three runs of version 1 (1.789 s to 1.907 s), and the medians are practically equal (1.795 s for v1 and 1.823 s for v2). This happens because:

- The imbalance of version 1 was already small: the slowest process was only about 7% above the mean, so that was the maximum possible gain.
- Version 2 adds work in rank 0 (reading the size of 3,000 files and sorting them); in run 1 the time outside the local count increased from about 0.33 s to 0.39 s.

As a complementary test, we also ran the provided `run_all.sh` with p = 8 (logs in `logs/run_all/`). In that execution every run of version 2 was faster than every run of version 1 (1.211 s vs 1.343 s on average, about 10% faster), with a speedup of 4.77 vs 4.30 and an average LI of 1.068 vs 1.136. This supports that version 2 does reduce the imbalance and can improve the total time, although the difference is small.

### c. Limitations

- **Few runs and high variability.** Only 3 runs per version, and the baseline varied 17% between runs (5.34 s to 6.27 s). Since the difference between v1 and v2 in the main experiment (about 6%) is smaller than this variation, it is not conclusive on its own. The speedup of about 3x over the sequential version is much larger than the noise, so that result remains reliable.
- **Results depend on the run conditions.** The `run_all.sh` execution was faster than the main experiment (speedup 4.30 to 4.77 instead of 3.10 to 3.30), probably because the dataset had just been generated inside the same container and the files were still in memory. This shows that file reading has a big impact on the measurements.
- **Single configuration.** Only p = 8 and one dataset (seed 42) were tested. With a more uneven dataset, the imbalance of version 1 would be larger and version 2 could show a bigger benefit.
- **Execution environment.** The experiment ran in Docker on a laptop: files are read through the shared folder of a virtual machine, the MacBook Air has no fan and can slow down when it heats up, and the M4 has two types of cores. These factors affect the local times and, therefore, the LI.
- **Static balancing only.** Version 2 balances bytes, not time. A process running on a slower core still takes longer, which a static split cannot correct. A dynamic distribution would be needed to evaluate this.

## 6. Conclusions

I would recommend version 2, the size-aware distribution. With p = 8 it reached a speedup of 3.30 and an efficiency of 0.41, compared to 3.10 and 0.39 for version 1, and it reduced the average LI from 1.068 to 1.047 by giving every process the same amount of data (34.38 MB each). The complementary `run_all.sh` execution showed the same trend more clearly: version 2 was faster in every run (speedup 4.77 vs 4.30) with a lower LI (1.068 vs 1.136). It is also almost as simple as version 1, since only the way the files are split changes. However, the improvement is small compared to the gain of parallelizing itself: both versions produce exactly the same result as the sequential baseline and are three to five times faster, and their efficiency stays between 0.39 and 0.60 mainly because of the non-parallel work and the competition for shared resources, not because of the load imbalance.