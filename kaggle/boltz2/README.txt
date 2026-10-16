Boltz-2 on Kaggle
1. Create a Kaggle Dataset named 'boltz-inputs' by uploading the folder dataset_boltz_inputs (data/, docking/, scripts/).
2. New Notebook -> File > Import Notebook -> boltz2_kaggle.ipynb.
3. Settings: Accelerator GPU, Internet On (Kaggle asks for phone verification once). Add the dataset boltz-inputs as input.
4. Run All. Set A (12 complexes) should take about 1-2 h on a T4; set B (80 complexes) a few more hours.
5. Download BOLTZ2_RESULTS.csv and BOLTZ2_ENRICHMENT.csv from the Output tab and put them in results/labgpu/.
