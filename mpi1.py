"""
MPI version 1: broadcast + static distribution of files.

- rank 0 reads consulta.txt and broadcasts the query words to everybody
- rank 0 lists the file_*.txt files and splits them in equal blocks
  (same number of files per process, without looking at their size)
- each process counts the query words in its own files
- rank 0 gathers the partial counters, joins them and prints the top 10
"""

import os
from collections import Counter

from mpi4py import MPI

from baseline_secuencial import cargar_consulta   # same reader as the baseline


def contar_en_archivos(archivos, dataset_dir, palabras_objetivo):
    """Counts the query words in a list of files (same logic as the baseline)."""
    freq = Counter()
    tokens = 0
    for fname in archivos:
        ruta = os.path.join(dataset_dir, fname)
        with open(ruta, "r", encoding="utf-8") as f:
            for linea in f:
                palabras = [w.lower() for w in linea.split()]
                tokens += len(palabras)
                for w in palabras:
                    if w in palabras_objetivo:
                        freq[w] += 1
    return freq, tokens


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir, "dataset")

    # everybody starts at the same time so the total time is fair
    comm.Barrier()
    t0 = MPI.Wtime()

    # rank 0 reads the query words
    palabras_objetivo = None
    if rank == 0:
        palabras_objetivo = cargar_consulta(os.path.join(dataset_dir, "consulta.txt"))

    # broadcast: all processes need the same query words
    palabras_objetivo = comm.bcast(palabras_objetivo, root=0)


    # rank 0 gets the list of files and makes one block per process

    bloques = None
    carga_bytes = None
    if rank == 0:
        archivos = sorted(f for f in os.listdir(dataset_dir)
                          if f.startswith("file_") and f.endswith(".txt"))
        # static block distribution: process i gets files [inicio, fin)
        bloques = []
        por_proceso = len(archivos) // size
        sobran = len(archivos) % size
        inicio = 0
        for i in range(size):
            fin = inicio + por_proceso + (1 if i < sobran else 0)
            bloques.append(archivos[inicio:fin])
            inicio = fin
        # only to report it: how many bytes ended up in each block
        carga_bytes = [sum(os.path.getsize(os.path.join(dataset_dir, f)) for f in b)
                       for b in bloques]
        

    # scatter: each process receives its block of files
    mis_archivos = comm.scatter(bloques, root=0)

    # local count, measuring only the local work
    t_local_inicio = MPI.Wtime()
    freq_local, tokens_local = contar_en_archivos(mis_archivos, dataset_dir, palabras_objetivo)
    t_local = MPI.Wtime() - t_local_inicio


    # gather the partial results in rank 0
    resumen_local = (freq_local, tokens_local, len(mis_archivos), t_local)
    todos = comm.gather(resumen_local, root=0)

    if rank == 0:
        # build the global result
        freq_global = Counter()
        tokens_totales = 0
        for freq, tokens, _, _ in todos:
            freq_global.update(freq)
            tokens_totales += tokens
        t_total = MPI.Wtime() - t0

        # save the global count with the same format as the baseline csv
        with open(os.path.join(dataset_dir, "mpi1_results.csv"), "w", encoding="utf-8") as f:
            f.write("palabra,conteo\n")
            for palabra in sorted(freq_global):
                f.write(f"{palabra},{freq_global[palabra]}\n")

        print(f"MPI version 1 (static block distribution) with {size} processes\n")
        print("Per-process report:")
        for r, (_, _, n_archivos, t) in enumerate(todos):
            print(f"  rank {r}: files={n_archivos}, MB={carga_bytes[r] / 1e6:.2f}, "
                  f"local_time={t:.6f} s")

        tiempos = [t for _, _, _, t in todos]
        promedio = sum(tiempos) / len(tiempos)
        li = max(tiempos) / promedio
        spread = (max(tiempos) - min(tiempos)) / promedio
        print(f"\nLoad imbalance LI = max/mean = {li:.3f}")
        print(f"Relative spread = (max-min)/mean = {spread:.3f}")


        print(f"\nTotal tokens read: {tokens_totales}")
        print(f"Total occurrences found: {sum(freq_global.values())}")
        print("\nTop 10 query words in the corpus:")
        for palabra, cuenta in freq_global.most_common(10):
            print(f"  {palabra}: {cuenta}")


        # this line is the one run_all.sh reads
        print(f"\nEXECUTION_TIME={t_total:.6f}")


if __name__ == "__main__":
    main()