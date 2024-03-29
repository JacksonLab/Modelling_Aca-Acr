# Aca-Acr regulatory circuit modelling

This repository contains the code used for stocastic modeling of the Aca-Acr regulatory circuit described in Birkholz *et al*. 2024: *Phage anti-CRISPR control by an RNA- and DNA-binding helix–turn–helix protein.*

The script [python aca_model_hpc.py] is called in parallel on an HPC platform using:

```bash
python aca_model_hpc.py --parameter [parameter_to_vary_for_analysis] --outdir [output_directory] --nsamples [number_of_simulations_to_run_per_thread] --threads [threads]
```

Where [parameter_to_vary_for_analysis] is one of:

| copy_number | replication_rate | transcription_rate | translation_rate | rna_kon | dna_kon | rna_decay | rna_kd | dna_kd |

Default paramter values and those varied for robustness analyses are hard-coded in the runscript.

## Citations

> Birkholz N, Kamata K, Feußner M, Wilkinson ME, Samaniego CC, Migur A, Kimanius D, Ceelen M, Went SC, Usher B, Blower TR, Brown CM, Beise CL, Weinberg Z, Fagerlund RD, Jackson SA and Fineran PC* (2024) Phage anti-CRISPR control by an RNA- and DNA-binding helix–turn–helix protein. *In revision*, **ISSUE**, Pstart–Pend. doi: [DOI](https://doi.org/doiheretoo)

